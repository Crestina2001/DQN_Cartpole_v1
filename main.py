from runner import train
from runner import test

ModeList=('Train','Test')
Mode=ModeList[1]
if Mode=='Train':
    train()
else:
    test(num=30, PATH='models/checkpoint600.pt')
