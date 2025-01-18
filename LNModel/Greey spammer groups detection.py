import ast
import csv


def greedy_spammer_group_selection(rf_csg, all_spammer):
    # Step 1: Initialize variables
    selected_groups = {}  # Dictionary to store detected spammer groups
    covered_spammers = set()
    remaining_spammers = set(all_spammer)
    print(len(remaining_spammers))
    # Step 2: Read and process the Struct_KG.txt file
    with open(rf_csg) as read_csg:
        next(read_csg)  # Skip the header
        candidate_groups = [ast.literal_eval(line) for line in read_csg]
        # print(candidate_groups)
    while remaining_spammers:
        best_key = None
        best_coverage = 0
        best_group = None

        # Step 3: Greedily select the key that covers the most uncovered spammers
        for csg in candidate_groups:
            for key, groups in csg.items():
                # Calculate coverage for this key
                current_coverage = sum(
                    1 for group in groups if (group[0] in remaining_spammers or group[1] in remaining_spammers))

                if current_coverage > best_coverage:
                    best_coverage = current_coverage
                    best_key = key
                    best_group = groups

        # Step 4: If no more coverage is possible, break the loop
        if best_coverage == 0:
            break

        # Step 5: Update selected groups and covered spammers
        if best_key not in selected_groups:
            selected_groups[best_key] = []

        for group in best_group:
            if group[0] in remaining_spammers or group[1] in remaining_spammers:
                selected_groups[best_key].append(group)
            if group[0] in remaining_spammers:
                remaining_spammers.remove(group[0])
            if group[1] in remaining_spammers:
                remaining_spammers.remove(group[1])
            covered_spammers.add(group[0])
            covered_spammers.add(group[1])

    return selected_groups


# Example usage:
# rf_csg = "../Data/Cell_Phones_and_Accessories/2014/Struct_KG.txt"
rf_csg = "../Data/YelpZip/Struct_KG.txt"
rf_cs = "../output/result spammers_YelpZip.csv"
detected_spammer_groups_file = "spammer_groups_YelpZip.txt"
fw = open(detected_spammer_groups_file, "w")
all_spammer = set()

# Assuming all_spammer is populated as before
with open(rf_cs) as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for row in reader:
        all_spammer.add(row[0])

detected_spammer_groups = greedy_spammer_group_selection(rf_csg, all_spammer)
fw.write(str(detected_spammer_groups))
# Output the detected spammer groups dictionary
print(detected_spammer_groups)
