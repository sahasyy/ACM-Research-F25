from scapy.all import PcapReader, TCP, IP
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import pickle
import os

PCAP_FILE = 'data/filtered.pcap'
MAX_PAYLOAD_LEN = 400  # bytes from paper

ATTACKER_IP = '172.16.0.1'
VICTIM_IP = '192.168.10.50'

# flow length limits for outliers
MIN_FLOW_LENGTH = 1      # remove flows <1 packet
MAX_FLOW_LENGTH = 230    # Remove flows >230 (picked by analysis)

# keep track of counts for each attack type
brute_count = 0
xss_count = 0
sqli_count = 0
normal_count = 0
filtered_count = 0  # count of flows removed due to length

# convert to unix time
def to_unix(dt_str):
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    return int(time.mktime(dt.timetuple()))

# define attack time windows on July 6, 2017 (from the dataset)
BRUTE_FORCE_START = to_unix("2017-07-06 07:20:00")
BRUTE_FORCE_END   = to_unix("2017-07-06 08:00:00")

XSS_START = to_unix("2017-07-06 08:15:00")
XSS_END   = to_unix("2017-07-06 08:35:00")

SQLI_START = to_unix("2017-07-06 08:40:00")
SQLI_END   = to_unix("2017-07-06 08:42:00")
CUTOFF_TIME = to_unix("2017-07-06 08:42:00")

# extract stuff from packer
def extract_packet_features(pkt):
    if not (pkt.haslayer(TCP) and pkt.haslayer(IP)):
        return None
    payload = bytes(pkt[TCP].payload)
    payload = payload[:MAX_PAYLOAD_LEN].ljust(MAX_PAYLOAD_LEN, b'\0')
    return {
        'timestamp': pkt.time,
        'src_ip': pkt[IP].src,
        'dst_ip': pkt[IP].dst,
        'src_port': pkt[TCP].sport,
        'dst_port': pkt[TCP].dport,
        'protocol': pkt[IP].proto,
        'payload': payload
    }

# read the file and convert packets to flows
def stream_packets_to_flows(pcap_file):
    flows = {}
    file_size = os.path.getsize(pcap_file)
    estimated_pkt_size = 300  # bytes (estimate)
    est_total_packets = file_size // estimated_pkt_size

    with PcapReader(pcap_file) as pcap:
        for pkt in tqdm(pcap, desc="Streaming and grouping packets", unit="packet", total=est_total_packets):
            features = extract_packet_features(pkt)
            if features is None:
                continue
            flow_key = (
                features['src_ip'],
                features['dst_ip'],
                features['src_port'],
                features['dst_port'],
                features['protocol']
            )
            if flow_key not in flows:
                flows[flow_key] = []
            flows[flow_key].append(features)

    return flows

# add labels to the flows by ip and attack windows
def label_for_flow(flow_key, flow_start_time):
    global brute_count, xss_count, sqli_count, normal_count
    src_ip, dst_ip, _, _, _ = flow_key

    if ((src_ip == ATTACKER_IP and dst_ip == VICTIM_IP) or
        (src_ip == VICTIM_IP and dst_ip == ATTACKER_IP)):
        # Removed the debug print to reduce spam

        if BRUTE_FORCE_START <= flow_start_time <= BRUTE_FORCE_END:
            brute_count += 1
            return 1
        elif XSS_START <= flow_start_time <= XSS_END:
            xss_count += 1
            return 2
        elif SQLI_START <= flow_start_time <= SQLI_END:
            sqli_count += 1
            return 3

    normal_count += 1
    return 0

# to find flow length cutoffs
def analyze_flow_lengths(flows):
    """Analyze flow length distribution to help set filtering thresholds"""
    lengths = [len(packets) for packets in flows.values()]
    lengths.sort()
    
    print(f"\nFlow Length Analysis:")
    print(f"Total flows: {len(lengths)}")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Mean length: {np.mean(lengths):.2f}")
    print(f"Median length: {np.median(lengths):.2f}")
    print(f"95th percentile: {np.percentile(lengths, 95):.0f}")
    print(f"99th percentile: {np.percentile(lengths, 99):.0f}")
    print(f"99.9th percentile: {np.percentile(lengths, 99.9):.0f}")
    
    # show how many flows would be filtered at different thresholds
    for threshold in [50, 100, 200, 500, 1000]:
        filtered = sum(1 for l in lengths if l > threshold)
        percentage = (filtered / len(lengths)) * 100
        print(f"Flows > {threshold} packets: {filtered} ({percentage:.2f}%)")
    
    return lengths

# convert flows to training data
def convert_flows_to_training_data(flows):
    global filtered_count
    X = []
    y = []
    lengths = []

    for flow_key, packets in tqdm(flows.items(), desc="Converting flows to training data", unit="flow"):
        packets.sort(key=lambda x: x['timestamp'])
        flow_start_time = int(packets[0]['timestamp'])

        if flow_start_time > CUTOFF_TIME:
            continue

        # filter out flows that are too short or too long
        flow_length = len(packets)
        if flow_length < MIN_FLOW_LENGTH or flow_length > MAX_FLOW_LENGTH:
            filtered_count += 1
            continue

        flow_payloads = [np.frombuffer(pkt['payload'], dtype=np.uint8).astype(np.float32) / 255.0
                         for pkt in packets]

        if not flow_payloads:
            continue

        X.append(np.stack(flow_payloads))
        y.append(label_for_flow(flow_key, flow_start_time))
        lengths.append(len(flow_payloads))

    return X, np.array(y), lengths

def suggest_max_flow_length(lengths):
    """Suggest a good MAX_FLOW_LENGTH based on data distribution"""
    lengths_array = np.array(lengths)
    
    # find a threshold that captures most flows but removes extreme outliers
    percentiles = [95, 98, 99, 99.5, 99.9]
    
    print(f"\n Suggested MAX_FLOW_LENGTH values:")
    for p in percentiles:
        threshold = int(np.percentile(lengths_array, p))
        flows_kept = sum(1 for l in lengths if l <= threshold)
        percentage_kept = (flows_kept / len(lengths)) * 100
        print(f"  {threshold:4d} packets (keeps {percentage_kept:.1f}% of flows)")
    
    # recommend 99th percentile as a good balance
    recommended = int(np.percentile(lengths_array, 99))
    print(f"\nRecommended: MAX_FLOW_LENGTH = {recommended}")
    return recommended

if __name__ == '__main__':
    print("Streaming packets with PcapReader...")
    flows = stream_packets_to_flows(PCAP_FILE)
    
    print("Analyzing flow length distribution...")
    flow_lengths = analyze_flow_lengths(flows)
    
    # suggest optimal flow length
    recommended_max = suggest_max_flow_length(flow_lengths)
    
    print(f"\n  CURRENT MAX_FLOW_LENGTH = {MAX_FLOW_LENGTH}")
    print(f"Consider updating to: MAX_FLOW_LENGTH = {recommended_max}")
    
    response = input(f"\nDo you want to use the recommended value ({recommended_max})? [y/N]: ")
    if response.lower() == 'y':
        MAX_FLOW_LENGTH = recommended_max
        print(f"Updated MAX_FLOW_LENGTH to {MAX_FLOW_LENGTH}")

    print("Saving raw flows to disk...")
    with open("raw_flows.pkl", "wb") as f:
        pickle.dump(flows, f)

    print("Converting to training data...")
    X, y, lengths = convert_flows_to_training_data(flows)

    print(f"\nProcessed {len(flows)} flows")
    print(f"Filtered out {filtered_count} flows (too short/long)")
    print(f"Final dataset: {len(X)} flows")
    
    if lengths:
        print(f"Flow lengths: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")
    
    # check class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_names = ['Normal', 'Brute Force', 'XSS', 'SQL Injection']
    print(f"\nClass distribution after filtering:")
    for class_id, count in zip(unique, counts):
        print(f"  {class_names[class_id]}: {count}")

    print("Saving training data to disk...")
    with open("X_varlen.pkl", "wb") as f:
        pickle.dump(X, f)
    with open("y_varlen.pkl", "wb") as f:
        pickle.dump(y, f)
    with open("lengths.pkl", "wb") as f:
        pickle.dump(lengths, f)

    print(f"\nFinal counts - Brute Force: {brute_count}, XSS: {xss_count}, SQLi: {sqli_count}, Normal: {normal_count}")
    print("Done!")