from random import shuffle

with open('Feature_MalwareGoodware_Weighted.csv','r') as ip:
    data=ip.readlines()


#for i in data:
#   print (i)

#print("\n \n \n")

shuffle(data)  # Shuffle old csv file
#for i in data:
#    print (i)

header, rest=data[0], data[1:]  # Devide Shuffled csv into two parts

with open('Shuffled_Feature_MalwareGoodware_Weighted.csv','w') as out:
    out.write(''.join([header]+rest))   # join Shuffled parts and create new csv file
