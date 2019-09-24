import pickle
import pandas as pd

# load complaint lables
with open("./one_h_lables.pickle", "rb") as myFile:
    labels = pickle.load(myFile)
print(type(labels))
print(len(labels))
print(labels[1])
print(len(labels[1]))

load mapping
with open("./complaint_category_index_map.pickle", "rb") as myFile:
    mapper = pickle.load(myFile)

print(type(mapper))
print(len(mapper))
print(mapper.items())

cat_list=[]
cat_list=mapper.keys()
for item in cat_list:
    print(item)

df=pd.DataFrame(cat_list,columns=['Category_List_Old'])
df.to_csv('category_list_old.csv',index=True)

