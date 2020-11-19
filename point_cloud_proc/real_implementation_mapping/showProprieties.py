import pickle
from tkinter import *
from tkinter import ttk
import time


def showGUI(prop):
    root = Tk() 
    root.title("Propriedades")
    tree = ttk.Treeview(root, selectmode ='browse',height = 20) 
    
    tree["columns"]=("value")
    tree.heading("value", text="Value")
    tree.column("value", width=900)

    color_list = []
    for key in prop: # key of proprety
        if(isinstance(prop[key], list)): # if is a list
            if((key == "planes") or (key =="cylinders") or (key =="secundaryplanes") or key =="cuboids"): # we need to load more if it is plane or cylinder
                arv = tree.insert("", 1, text=key, open=True) # add major tree primitive division
                for o in range(len(prop[key])): # Repeat for every object
                    colorzinha = prop[key][o]["color"]
                    color_list.append(colorzinha)
                    arv2 = tree.insert(arv, 2, text=(key+" "+str(o)), tags=[str(colorzinha)]) # add plane 1, plane 2, plane 3 ....
                    for key2 in prop[key][o]: # iterate over propriety
                        print("INSERIU")
                        tree.insert(arv2, "end", text=(key2), values=(str(prop[key][o][key2]),))
            else:
                arv = tree.insert("", 1, text=key, values=(str(prop[key]),))
        else:
            arv = tree.insert("", 1, text=key, values=(str(prop[key]),))

    for cor in color_list:
        mycolor = '#%02x%02x%02x' % (int(cor[0]*255), int(cor[1]*255), int(cor[2]*255))
        tree.tag_configure(str(cor), background=mycolor)
    tree.pack(fill='x')
    root.mainloop()

while True:
    f = open('feat.pckl', 'rb')
    obj = pickle.load(f)
    f.close()
    print(obj)
    showGUI(obj)
    time.sleep(0.1)
    


