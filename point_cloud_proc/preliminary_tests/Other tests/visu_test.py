from tkinter import *
from tkinter import ttk

root = Tk()

tree = ttk.Treeview(root)

tree["columns"]=("value")
# tree.column("one", width=100 )
# tree.column("two", width=100)
tree.heading("value", text="Value")
#tree.heading("value", text="column B")

#tree.insert("" , 0,    text="Line 1", values=("1A"))

id2 = tree.insert("", 1, "mainplanes", text="Main Planes")
p1 = tree.insert(id2, 2, "plane1", text="Plane 1")
tree.insert(p1, "end", "size", text="Size")

##alternatively:
tree.insert("", 3, "dir3", text="Dir 3")
tree.insert("dir3", 3, text=" sub dir 3",values=("3A"))

tree.pack()
root.mainloop()