import csv

rf_csg = "../Data/Yelp_New York/Struct_KG.txt"
rf_cs = "../output/result spammers_YelpNYC.csv"
detected_spammer_groups_file = "spammer_groups_YelpNYC.txt"
all_spammer = []
spammer_group = {}
with open(rf_cs) as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for row in reader:
        all_spammer.append(row[0])

with open(rf_csg) as read_csg:
    # read candidate spammer groups file
    next(read_csg)
    for line in read_csg:
        csg = eval(line)
        for key in csg.keys():
            spammer_group[key] = []
            for group in csg[key]:
                if group[0] or group[1] in all_spammer:
                    spammer_group[key].append(group)
                # if group[0] in all_spammer:
                #     spammer_group[key].append((group[0], group[2], group[4]))
                # if group[1] in all_spammer:
                #     spammer_group[key].append((group[1], group[3], group[5]))
for key in spammer_group.keys():
    if len(spammer_group[key]) != 0:
        print(key, spammer_group[key])
print(len(spammer_group.keys()))
fw = open(detected_spammer_groups_file, "w")

fw.write(str(spammer_group))
