# Data Card: NSL-KDD Dataset

## Dataset Information

### Dataset Name
NSL-KDD (Network Security Laboratory - Knowledge Discovery and Data Mining)

### Purpose
The NSL-KDD dataset is designed for network intrusion detection research. It is an improved version of the KDD Cup 99 dataset, addressing issues such as redundant records and unrealistic data distributions that were present in the original dataset.

### Authors
- **Original KDD Cup 99**: DARPA Intrusion Detection Evaluation Program
- **NSL-KDD Improvement**: University of New Brunswick, Canadian Institute for Cybersecurity

### Dataset Owner
University of New Brunswick, Canadian Institute for Cybersecurity

### License
The NSL-KDD dataset is publicly available for research purposes. The dataset is provided free of charge for academic and research use.

**Source**: [University of New Brunswick - NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)

## Dataset Characteristics

### Size
- **Training Set**: ~125,000 records (KDDTrain+.txt)
- **Test Set**: ~22,500 records (KDDTest+.txt)
- **Features**: 41 features per record
- **Classes**: Binary (Normal vs. Anomaly), with sub-classes for attack types (DoS, Probe, R2L, U2R)

### Features
The dataset contains 41 features representing network connection characteristics:

**Basic Features (9 features):**
- duration, protocol_type, service, flag, src_bytes, dst_bytes, land, wrong_fragment, urgent

**Content Features (13 features):**
- hot, num_failed_logins, logged_in, num_compromised, root_shell, su_attempted, num_root, num_file_creations, num_shells, num_access_files, num_outbound_cmds, is_host_login, is_guest_login

**Traffic Features (9 features):**
- count, srv_count, serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate

**Host-based Traffic Features (10 features):**
- dst_host_count, dst_host_srv_count, dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rate, dst_host_srv_diff_host_rate, dst_host_serror_rate, dst_host_srv_serror_rate, dst_host_rerror_rate, dst_host_srv_rerror_rate

### Labels
- **Normal**: Legitimate network traffic
- **Anomaly/Attack Types**:
  - **DoS (Denial of Service)**: Flooding attacks
  - **Probe**: Surveillance and scanning attacks
  - **R2L (Remote to Local)**: Unauthorized access attempts
  - **U2R (User to Root)**: Privilege escalation attempts

## Data Processing

### Preprocessing Steps Applied

1. **Categorical Encoding**:
   - `protocol_type`: Encoded using label encoding (TCP, UDP, ICMP, etc.)
   - `service`: Encoded using label encoding (HTTP, FTP, SMTP, etc.)
   - `flag`: Encoded using label encoding (SF, S0, REJ, etc.)

2. **Feature Scaling**:
   - All numerical features are standardized using `StandardScaler` from scikit-learn
   - Mean = 0, Standard Deviation = 1

3. **Missing Value Handling**:
   - NSL-KDD is pre-cleaned and contains no missing values
   - Any potential missing values are replaced with 0

4. **Data Splitting**:
   - **Training Set**: 70% of data
   - **Validation Set**: 10% of data (from training split)
   - **Test Set**: 20% of data
   - Stratified splitting to maintain class distribution

5. **Data Cleaning**:
   - Removal of exceptional cases (outliers beyond 3 standard deviations)
   - Handling of infinite values (replaced with 0)
   - Normalization of feature ranges

### Data Quality

- **Completeness**: High - dataset is pre-processed and complete
- **Consistency**: High - consistent feature encoding and formats
- **Accuracy**: High - labels are verified and accurate
- **Timeliness**: Dataset represents network traffic from 1999 DARPA evaluation, may not reflect modern traffic patterns

### Limitations

1. **Temporal Relevance**: Dataset is based on 1999 network traffic, which may not represent modern attack patterns
2. **Feature Engineering**: Some features may be outdated for modern network environments
3. **Class Imbalance**: Significant imbalance between normal and attack classes, and between attack types
4. **Synthetic Nature**: Some attack types (especially U2R and R2L) have very few samples

### Ethical Considerations

- **Privacy**: Network traffic data may contain sensitive information, but NSL-KDD is anonymized
- **Bias**: Dataset may contain biases from the original data collection period
- **Use Case**: Intended for research and educational purposes in cybersecurity

## Dataset Access

### Download Location
- **Primary Source**: https://www.unb.ca/cic/datasets/nsl.html
- **Alternative**: Various academic repositories and research datasets collections

### Usage Notes
- Dataset is publicly available for research purposes
- No special permissions required for academic use
- Commercial use may require additional permissions

### Citation
If using NSL-KDD in research, please cite:

```
M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, 
"A Detailed Analysis of the KDD CUP 99 Data Set," 
Proceedings of the Second IEEE Symposium on Computational 
Intelligence for Security and Defense Applications (CISDA), 2009.
```

## Data Version

- **Version Used**: NSL-KDD (improved version of KDD Cup 99)
- **Date Accessed**: 2024
- **Processing Version**: Custom preprocessing pipeline as described above
