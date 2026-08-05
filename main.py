import math
import random


# ============================================================
# MACHINE LEARNING-BASED CAN BUS INTRUSION DETECTION
# Industrial IoT - STM32H7 / CAN Bus
# ============================================================


# ------------------------------------------------------------
# CAN FRAME
# ------------------------------------------------------------

class CANFrame:

    def __init__(self, timestamp, can_id, data):
        self.timestamp = timestamp
        self.can_id = can_id
        self.data = data
        self.dlc = len(data)

    def payload_mean(self):

        if self.dlc == 0:
            return 0

        return sum(self.data) / self.dlc

    def payload_std(self):

        if self.dlc == 0:
            return 0

        mean = self.payload_mean()

        total = 0

        for value in self.data:
            total += (value - mean) ** 2

        return math.sqrt(total / self.dlc)

    def payload_sum(self):

        return sum(self.data)

    def nonzero_bytes(self):

        count = 0

        for value in self.data:
            if value != 0:
                count += 1

        return count


# ------------------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------------------

def extract_features(frame, previous_frame):

    if previous_frame is None:

        inter_arrival = 0.0

    else:

        inter_arrival = (
            frame.timestamp -
            previous_frame.timestamp
        )

    if inter_arrival <= 0:
        message_rate = 0.0

    else:
        message_rate = 1.0 / inter_arrival

    features = [

        float(frame.can_id),

        float(frame.dlc),

        frame.payload_mean(),

        frame.payload_std(),

        float(frame.payload_sum()),

        float(frame.nonzero_bytes()),

        inter_arrival,

        message_rate

    ]

    return features


# ------------------------------------------------------------
# EUCLIDEAN DISTANCE
# ------------------------------------------------------------

def distance(a, b):

    total = 0.0

    for i in range(len(a)):

        difference = a[i] - b[i]

        total += difference * difference

    return math.sqrt(total)


# ------------------------------------------------------------
# NORMALIZE FEATURES
# ------------------------------------------------------------

def calculate_statistics(dataset):

    feature_count = len(dataset[0])

    means = [0.0] * feature_count
    stds = [0.0] * feature_count

    for row in dataset:

        for i in range(feature_count):

            means[i] += row[i]

    for i in range(feature_count):

        means[i] /= len(dataset)

    for row in dataset:

        for i in range(feature_count):

            difference = row[i] - means[i]

            stds[i] += difference * difference

    for i in range(feature_count):

        stds[i] = math.sqrt(
            stds[i] / len(dataset)
        )

        if stds[i] == 0:

            stds[i] = 1.0

    return means, stds


def normalize(row, means, stds):

    result = []

    for i in range(len(row)):

        value = (
            row[i] - means[i]
        ) / stds[i]

        result.append(value)

    return result


# ------------------------------------------------------------
# K-NEAREST NEIGHBOR ANOMALY DETECTION
# ------------------------------------------------------------

def anomaly_score(
        sample,
        normal_data,
        k=3):

    distances = []

    for normal_sample in normal_data:

        d = distance(
            sample,
            normal_sample
        )

        distances.append(d)

    distances.sort()

    k = min(k, len(distances))

    if k == 0:
        return 0.0

    total = 0.0

    for i in range(k):

        total += distances[i]

    return total / k


# ------------------------------------------------------------
# CREATE NORMAL CAN TRAFFIC
# ------------------------------------------------------------

def generate_normal_frames():

    frames = []

    timestamp = 0.0

    for i in range(40):

        timestamp += 0.010

        data = [

            20 + random.randint(-2, 2),
            40 + random.randint(-2, 2),
            60 + random.randint(-2, 2),
            80 + random.randint(-2, 2),
            100 + random.randint(-2, 2),
            120 + random.randint(-2, 2),
            140 + random.randint(-2, 2),
            160 + random.randint(-2, 2)

        ]

        frame = CANFrame(
            timestamp,
            0x120,
            data
        )

        frames.append(frame)

    return frames


# ------------------------------------------------------------
# CREATE ATTACK CAN TRAFFIC
# ------------------------------------------------------------

def generate_attack_frames():

    frames = []

    timestamp = 0.0

    for i in range(10):

        timestamp += 0.001

        data = [

            255,
            255,
            255,
            255,
            255,
            255,
            255,
            255

        ]

        frame = CANFrame(
            timestamp,
            0x7FF,
            data
        )

        frames.append(frame)

    return frames


# ------------------------------------------------------------
# BUILD FEATURE DATASET
# ------------------------------------------------------------

def build_features(frames):

    dataset = []

    previous = None

    for frame in frames:

        features = extract_features(
            frame,
            previous
        )

        dataset.append(features)

        previous = frame

    return dataset


# ------------------------------------------------------------
# DETECT INTRUSIONS
# ------------------------------------------------------------

def detect_intrusions(
        frames,
        normal_model,
        means,
        stds,
        threshold):

    predictions = []

    previous = None

    for frame in frames:

        features = extract_features(
            frame,
            previous
        )

        normalized = normalize(
            features,
            means,
            stds
        )

        score = anomaly_score(
            normalized,
            normal_model
        )

        if score > threshold:

            prediction = 1

        else:

            prediction = 0

        predictions.append(
            (
                frame,
                score,
                prediction
            )
        )

        previous = frame

    return predictions


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

def confusion_matrix(
        actual,
        predicted):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(actual)):

        if actual[i] == 1 and predicted[i] == 1:
            tp += 1

        elif actual[i] == 0 and predicted[i] == 0:
            tn += 1

        elif actual[i] == 0 and predicted[i] == 1:
            fp += 1

        elif actual[i] == 1 and predicted[i] == 0:
            fn += 1

    return tp, tn, fp, fn


# ------------------------------------------------------------
# PERFORMANCE METRICS
# ------------------------------------------------------------

def calculate_metrics(
        actual,
        predicted):

    tp, tn, fp, fn = confusion_matrix(
        actual,
        predicted
    )

    total = tp + tn + fp + fn

    if total == 0:
        accuracy = 0

    else:
        accuracy = (
            (tp + tn) / total
        )

    if tp + fp == 0:
        precision = 0

    else:
        precision = (
            tp / (tp + fp)
        )

    if tp + fn == 0:
        recall = 0

    else:
        recall = (
            tp / (tp + fn)
        )

    if precision + recall == 0:
        f1 = 0

    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    return (
        tp,
        tn,
        fp,
        fn,
        accuracy,
        precision,
        recall,
        f1
    )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print()
    print("=" * 65)
    print(" MACHINE LEARNING-BASED CAN BUS")
    print(" INTRUSION DETECTION FOR INDUSTRIAL IoT")
    print("=" * 65)

    print()
    print("System Architecture:")
    print()
    print("STM32H7")
    print("   |")
    print("   v")
    print("FDCAN Peripheral")
    print("   |")
    print("   v")
    print("CAN Transceiver")
    print("   |")
    print("   v")
    print("CAN Bus")
    print("   |")
    print("   v")
    print("ML Intrusion Detection")
    print()

    # --------------------------------------------------------
    # GENERATE NORMAL TRAFFIC
    # --------------------------------------------------------

    random.seed(42)

    normal_frames = (
        generate_normal_frames()
    )

    print(
        "Normal CAN Frames Generated :",
        len(normal_frames)
    )

    # --------------------------------------------------------
    # BUILD TRAINING DATA
    # --------------------------------------------------------

    normal_features = build_features(
        normal_frames
    )

    means, stds = calculate_statistics(
        normal_features
    )

    normal_model = []

    for row in normal_features:

        normalized = normalize(
            row,
            means,
            stds
        )

        normal_model.append(
            normalized
        )

    print(
        "Normal Training Samples      :",
        len(normal_model)
    )

    # --------------------------------------------------------
    # CALCULATE ANOMALY THRESHOLD
    # --------------------------------------------------------

    training_scores = []

    for sample in normal_model:

        score = anomaly_score(
            sample,
            normal_model,
            3
        )

        training_scores.append(
            score
        )

    maximum_training_score = max(
        training_scores
    )

    threshold = (
        maximum_training_score * 1.5
    )

    print(
        "Anomaly Detection Threshold  :",
        round(threshold, 4)
    )

    # --------------------------------------------------------
    # CREATE TEST DATA
    # --------------------------------------------------------

    test_normal = generate_normal_frames()

    test_attack = generate_attack_frames()

    test_frames = (
        test_normal +
        test_attack
    )

    actual_labels = []

    for i in range(len(test_normal)):
        actual_labels.append(0)

    for i in range(len(test_attack)):
        actual_labels.append(1)

    # --------------------------------------------------------
    # DETECTION
    # --------------------------------------------------------

    results = detect_intrusions(
        test_frames,
        normal_model,
        means,
        stds,
        threshold
    )

    predicted_labels = []

    print()
    print("=" * 65)
    print(" CAN BUS INTRUSION DETECTION RESULTS")
    print("=" * 65)

    print()

    for i in range(len(results)):

        frame, score, prediction = (
            results[i]
        )

        predicted_labels.append(
            prediction
        )

        if prediction == 1:

            status = "INTRUSION"

        else:

            status = "NORMAL"

        print(
            "Frame {:02d} | CAN ID: 0x{:03X} | "
            "DLC: {} | Score: {:.3f} | {}"
            .format(
                i + 1,
                frame.can_id,
                frame.dlc,
                score,
                status
            )
        )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    (
        tp,
        tn,
        fp,
        fn,
        accuracy,
        precision,
        recall,
        f1
    ) = calculate_metrics(
        actual_labels,
        predicted_labels
    )

    print()
    print("=" * 65)
    print(" PERFORMANCE EVALUATION")
    print("=" * 65)

    print()
    print("True Positives  :", tp)
    print("True Negatives  :", tn)
    print("False Positives :", fp)
    print("False Negatives :", fn)

    print()
    print(
        "Accuracy        : {:.2f}%"
        .format(accuracy * 100)
    )

    print(
        "Precision       : {:.2f}%"
        .format(precision * 100)
    )

    print(
        "Recall          : {:.2f}%"
        .format(recall * 100)
    )

    print(
        "F1-Score        : {:.2f}%"
        .format(f1 * 100)
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print(" FINAL SYSTEM STATUS")
    print("=" * 65)

    attack_count = 0

    for prediction in predicted_labels:

        if prediction == 1:
            attack_count += 1

    print()
    print(
        "Total CAN Frames Analysed :",
        len(test_frames)
    )

    print(
        "Intrusions Detected       :",
        attack_count
    )

    if attack_count > 0:

        print()
        print(
            "[ALERT] Suspicious CAN bus activity detected!"
        )

    else:

        print()
        print(
            "[SAFE] CAN bus traffic appears normal."
        )

    print()
    print("=" * 65)


main()