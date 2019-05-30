from pyspark import SparkContext
from scapy.all import *
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import csv
import sys


def preview(pkt):
    # We create a line in this function - contains all features' values
    if(pkt.getlayer(Ether)):
        # eth_src = pkt.getlayer(Ether).src
        # eth_dst = pkt.getlayer(Ether).dst
        eth_type = pkt.getlayer(Ether).type
    else:
        # eth_src = 0
        # eth_dst = 0
        eth_type = 0

    if(pkt.getlayer(IP)):
        ip_version = pkt.getlayer(IP).version
        ip_ihl = pkt.getlayer(IP).ihl
        ip_tos = pkt.getlayer(IP).tos
        ip_len = pkt.getlayer(IP).len
        ip_id = pkt.getlayer(IP).id
        ip_frag = pkt.getlayer(IP).frag
        ip_ttl = pkt.getlayer(IP).ttl
        ip_proto = pkt.getlayer(IP).proto
        ip_chksum = pkt.getlayer(IP).chksum
        # ip_src = pkt.getlayer(IP).src
        # ip_dst = pkt.getlayer(IP).dst
    else:
        ip_version = 4
        ip_ihl = 0
        ip_tos = 0
        ip_len = 0
        ip_id = 1
        ip_frag = 0
        ip_ttl = 64
        ip_proto = 0
        ip_chksum = 0
        # ip_src = 0
        # ip_dst = 0

    if(pkt.getlayer(TCP)):
        tcp_sport = pkt.getlayer(TCP).sport
        tcp_dport = pkt.getlayer(TCP).dport
        tcp_seq = pkt.getlayer(TCP).seq
        tcp_ack = pkt.getlayer(TCP).ack
        tcp_dataofs = pkt.getlayer(TCP).dataofs
        tcp_reserved = pkt.getlayer(TCP).reserved
        tcp_window = pkt.getlayer(TCP).window
        tcp_chksum = pkt.getlayer(TCP).chksum
        tcp_urgptr = pkt.getlayer(TCP).urgptr
    else:
        tcp_sport = 20
        tcp_dport = 80
        tcp_seq = 0
        tcp_ack = 0
        tcp_dataofs = 0
        tcp_reserved = 0
        tcp_window = 8192
        tcp_chksum = 0
        tcp_urgptr = 0

    if(pkt.getlayer(IPv6)):
        ipv6_version = pkt.getlayer(IPv6).version
        ipv6_tc = pkt.getlayer(IPv6).tc
        ipv6_fl = pkt.getlayer(IPv6).fl
        ipv6_plen = pkt.getlayer(IPv6).plen
        ipv6_nh = pkt.getlayer(IPv6).nh
        ipv6_hlim = pkt.getlayer(IPv6).hlim
        # ipv6_src = pkt.getlayer(IPv6).src
        # ipv6_dst = pkt.getlayer(IPv6).dst
    else:
        ipv6_version = 6
        ipv6_tc = 0
        ipv6_fl = 0
        ipv6_plen = 0
        ipv6_nh = 59
        ipv6_hlim = 64
        # ipv6_src = 0
        # ipv6_dst = 0

    if(pkt.getlayer(UDP)):
        udp_sport = pkt.getlayer(UDP).sport
        udp_dport = pkt.getlayer(UDP).dport
        udp_len = pkt.getlayer(UDP).len
        udp_chksum = pkt.getlayer(UDP).chksum
    else:
        udp_sport = 53
        udp_dport = 53
        udp_len = 0
        udp_chksum = 0

    if(pkt.getlayer(ICMP)):
        icmp_type = pkt.getlayer(ICMP).type
        icmp_code = pkt.getlayer(ICMP).code
        icmp_chksum = pkt.getlayer(ICMP).chksum
        icmp_id = pkt.getlayer(ICMP).id
        icmp_seq = pkt.getlayer(ICMP).seq
        icmp_ts_ori = pkt.getlayer(ICMP).ts_ori
        icmp_ts_rx = pkt.getlayer(ICMP).ts_rx
        icmp_ts_tx = pkt.getlayer(ICMP).ts_tx
        icmp_ptr = pkt.getlayer(ICMP).ptr
        icmp_reserved = pkt.getlayer(ICMP).reserved
        icmp_length = pkt.getlayer(ICMP).length
        icmp_nexthopmtu = pkt.getlayer(ICMP).nexthopmtu
    else:
        icmp_type = 8
        icmp_code = 0
        icmp_chksum = 0
        icmp_id = 0
        icmp_seq = 0
        icmp_ts_ori = 54305877
        icmp_ts_rx = 54305877
        icmp_ts_tx = 54305877
        icmp_ptr = 0
        icmp_reserved = 0
        icmp_length = 0
        icmp_nexthopmtu = 0

    if(pkt.getlayer(ARP)):
        arp_hwtype = pkt.getlayer(ARP).hwtype
        arp_ptype = pkt.getlayer(ARP).ptype
        arp_hwlen = pkt.getlayer(ARP).hwlen
        arp_plen = pkt.getlayer(ARP).plen
        arp_op = pkt.getlayer(ARP).op
        # arp_hwsrc = pkt.getlayer(ARP).hwsrc
        # arp_psrc = pkt.getlayer(ARP).psrc
        # arp_hwdst = pkt.getlayer(ARP).hwdst
        # arp_pdst = pkt.getlayer(ARP).pdst
    else:
        arp_hwtype = 1
        arp_ptype = 2048
        arp_hwlen = 0
        arp_plen = 0
        arp_op = 1
        # arp_hwsrc = 0
        # arp_psrc = 0
        # arp_hwdst = 0
        # arp_pdst = 0

    global cnt
    global datasetLength
    global labels
    label_ = labels[cnt]  
    if label_ == "BENIGN":
        label = 0
    else:
        label = 1
    cnt = cnt + 1
    print float("{0:.2f}".format((cnt/float(datasetLength))*100))
    writer.writerow([eth_type,
                     ip_version, ip_ihl, ip_tos, ip_len, ip_id,
                     ip_frag, ip_ttl, ip_proto, ip_chksum, tcp_sport, tcp_dport, tcp_seq, tcp_ack,
                     tcp_dataofs, tcp_reserved, tcp_window, tcp_chksum,
                     tcp_urgptr, ipv6_version, ipv6_tc, ipv6_fl, ipv6_plen,
                     ipv6_nh, ipv6_hlim, udp_sport,
                     udp_dport, udp_len, udp_chksum, icmp_type, icmp_code,
                     icmp_chksum, icmp_id, icmp_seq, icmp_ts_ori, icmp_ts_rx,
                     icmp_ts_tx, icmp_ptr, icmp_reserved, icmp_length, icmp_nexthopmtu,
                     arp_hwtype, arp_ptype, arp_hwlen, arp_plen, arp_op, label])


# day of week
day = str(sys.argv[1])
labelsFile = str(sys.argv[2])
pcapFile = str(sys.argv[3])
# each SparkContext creates its own Spark application
sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
df = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load(labelsFile)
# get the total size of the csv with the labels - same size with pcap file
baseRDD = df.rdd
datasetLength = baseRDD.count()
# get only the labels
labels = baseRDD.map(lambda x: x[-1]).take(datasetLength)
# print len(labels)
# print datasetLength
cnt = 0

with open(day + 'LabeledPcap.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    sniff(offline=pcapFile, prn=preview, store=0, count=datasetLength)
csvFile.close()
