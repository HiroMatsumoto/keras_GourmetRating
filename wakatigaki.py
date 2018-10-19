import sys
argvs = sys.argv
import MeCab
m = MeCab.Tagger("-Owakati")
filepath = './rating4.csv'
text = open(filepath,"r",encoding = "shift-jis")
owakatifile = open('wakatigakidone4.csv',"w",encoding="shift-jis")

for line in text:
    result = m.parse(line)
    owakatifile.writelines(result)

owakatifile.flush()
owakatifile.close()
