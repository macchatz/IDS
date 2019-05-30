1. Create a cluster with at least #2 of instances and send the sparkfiles directory to the master node

2. When connected in the cluster's master install the following packages/libraries:
    sudo yum update
    sudo yum install python-pip
    sudo pip install scapy
    sudo pip install pandas
    sudo pip install sklearn

3. Run the following commands to create a dataset:
    a.(get the features)  wget https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/PCAPs/
    b.(get the labels)    wget https://iscxdownloads.cs.unb.ca/iscxdownloads/CIC-IDS-2017/MachineLearningCSV.zip
    c.(run the preprocessing.py script for each day)
        spark/bin/spark-submit --packages com.databricks:spark-csv_2.10:1.2.0 preprocessing.py Monday /home/ec2-user/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv /home/ec2-user/Monday-WorkingHours.pcap
        spark/bin/spark-submit --packages com.databricks:spark-csv_2.10:1.2.0 preprocessing.py Tuesday /home/ec2-user/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv /home/ec2-user/Tuesday-WorkingHours.pcap
        spark/bin/spark-submit --packages com.databricks:spark-csv_2.10:1.2.0 preprocessing.py Wednesday /home/ec2-user/MachineLearningCVE/Wednesday-WorkingHours.pcap_ISCX.csv /home/ec2-user/Wednesday-WorkingHours.pcap
        spark/bin/spark-submit --packages com.databricks:spark-csv_2.10:1.2.0 preprocessing.py Thursday /home/ec2-user/MachineLearningCVE/Thursday-WorkingHours.pcap_ISCX.csv /home/ec2-user/Thursday-WorkingHours.pcap
        spark/bin/spark-submit --packages com.databricks:spark-csv_2.10:1.2.0 preprocessing.py Friday /home/ec2-user/MachineLearningCVE/Friday-WorkingHours.pcap_ISCX.csv /home/ec2-user/Friday-WorkingHours.pcap
    d.(concatenate all csv per day into a single csv)
        cat MondayLabeledPcap.csv TuesdayLabeledPcap.csv WednesdayLabeledPcap.csv ThursdayLabeledPcap.csv FridayLabeledPcap.csv > LabeledNetworkTraffic.csv

4. Run the following command to run the machineLearning.py script on the processed data
    spark/bin/spark-submit machineLearning.py