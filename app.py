

#######################libraries#######################################################
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import colorama
import re
import pandas as pd
import mido
from time import sleep
import enum
import numpy as np
import random as random



#######################################Neural Network Model###############################
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

tokens = tokenizer.encode('It was good but couldve been better. Great', return_tensors='pt')
result = model(tokens)
result.logits
int(torch.argmax(result.logits))+1
result



#######################################Web scraping poems########################################


url = "http://www.versedaily.org/archives.shtml"
urls = set()
req = requests.get(url)
soup = BeautifulSoup(req.content, 'html.parser')
for link in soup.find_all('a') : 
    urls.add(link.get('href'))


data = {"titles" : [], "poems" : []}
# i = 0
for url in list(urls)[:15]: 
  # print(i)
  req = requests.get(url)
  soup = BeautifulSoup(req.content, 'html.parser')
  poem = []
  for para in soup.find_all('font', face="Times New Roman,Times") : 
    poem.append(para.get_text())
  try : 
    poem = poem[1:-2]
    a = re.search(r'\b(Tweet)\b', poem[-1])
    poem[-1] = poem[-1][:a.start() - 4]  # detect 1ere iteration de tweet puis supprimer la suite 
    if len(poem) == 2:
      poem[1] = poem[1].replace('\n', ' ').replace("\r", ' ').replace('\xa0', " ")
      data["titles"].append(poem[0])
      data["poems"].append(poem[1])
  except:
    continue
  # i = i + 1


data = pd.DataFrame(data)



data.loc[2,"poems"]


###########################################pop song###################################
data = {"titles" : [], "poems" : []}

data["titles"].append('still standing')
data["titles"].append("Yesterday")

data["poems"].append("You could never know what it s like +\
Your blood like winter freezes just like ice +\
And there s a cold lonely light that shines from you +\
You ll wind up like the wreck you hide behind that mask you use +\
And did you think this fool could never win? +\
Well look at me, I m coming back again +\
I got a taste of love in a simple way +\
And if you need to know while I m still standing, you just fade away +\
Don t you know I m still standing better than I ever did +\
Looking like a true survivor, feeling like a little kid +\
I m still standing after all this time +\
Picking up the pieces of my life without you on my mind +\
I m still standing (Yeah, yeah, yeah) +\
I m still standing (Yeah, yeah, yeah) +\
Once I never could have hoped to win +\
You re starting down the road leaving me again +\
The threats you made were meant to cut me down +\
And if our love was just a circus, you d be a clown by now +\
You know I m still standing better than I ever did +\
Looking like a true survivor, feeling like a little kid +\
I m still standing after all this time +\
Picking up the pieces of my life without you on my mind +\
I m still standing (Yeah, yeah, yeah) +\
I m still standing (Yeah, yeah, yeah) +\
Don t you know that I m still standing better than I ever did +\
Looking like a true survivor, feeling like a little kid +\
I m still standing after all this time +\
Picking up the pieces of my life without you on my mind +\
I m still standing Yeah, yeah, yeah +\
I m still standing Yeah, yeah, yeah +\
I m still standing Yeah, yeah, yeah +\
I m still standing Yeah, yeah, yeah +\
I m still standing Yeah, yeah, yeah +\
I m still standing Yeah, yeah, yeah")

data["poems"].append("Yesterday +\
All my troubles seemed so far away +\
Now it looks as though they're here to stay +\
Oh, I believe in yesterday +\
Suddenly +\
I'm not half the man I used to be +\
There's a shadow hangin' over me +\
Oh, yesterday came suddenly +\
Why she had to go, I don't know, she wouldn't say +\
I said something wrong, now I long for yesterday +\
Yesterday +\
Love was such an easy game to play +\
Now I need a place to hide away +\
Oh, I believe in yesterday +\
Why she had to go, I don't know, she wouldn't say +\
I said something wrong, now I long for yesterday +\
Yesterday +\
Love was such an easy game to play +\
Now I need a place to hide away +\
Oh, I believe in yesterday")

data = pd.DataFrame(data)
##########################################music synthetiser###############################



class chord_type(enum.Enum):
    Major=1
    Minor=2
    Augmented=3
    Diminished=4

class strum_pattern(enum.Enum):
    DDUUDU=1
    DDUUDD=2
    DDU=3
    DDUDUDUDUDU=4
    DDDD=5

#region : Enumerating Notes
A = 57
A_sharp=58
B=59
C=60
C_sharp=61
D=62
D_sharp=63
E=64
F=65
F_sharp=66
G=67
G_sharp=68
#endregion

outport = mido.open_output()    



def note(note, velocity=64, time=2):

    return mido.Message('note_on', note=note, velocity=velocity, time=time)

def note_off(note, velocity=64, time=2):

   return mido.Message('note_off', note=note, velocity=velocity, time=time)

def pause():

    sleep(.01)

def Chord(root,duration,is_natural_sounding,is_down,chord_type):


    if chord_type==chord_type.Major:
        a =4
        b= 7
    elif chord_type==chord_type.Minor:
        a=3
        b=7
    elif chord_type==chord_type.Augmented:
        a=4
        b=8
    elif chord_type==chord_type.Diminished:
        a=3
        b=6

    if is_down==True:

        outport.send(note(root -12+a))
        if is_natural_sounding == True: pause()
        outport.send(note(root -12+b))
        if is_natural_sounding==True: pause()
        outport.send(note(root))
        if is_natural_sounding == True: pause()
        outport.send(note(root+a))
        if is_natural_sounding == True: pause()
        outport.send(note(root+b))

        sleep(duration)

        outport.send(note_off(root-12+a))
        outport.send(note_off(root-12+b))
        outport.send(note_off(root))
        outport.send(note_off(root+a))
        outport.send(note_off(root+b))
        outport.send(note_off(root + b))

    else:
        outport.send(note(root,54))
        if is_natural_sounding == True: pause()
        outport.send(note(root+a,54))
        if is_natural_sounding == True: pause()
        outport.send(note(root+b,54))
        if is_natural_sounding == True: pause()
        outport.send(note(root -12+b,54))
        if is_natural_sounding == True: pause()
        outport.send(note(root - 12 + a,54))


        sleep(duration)

        outport.send(note_off(root ))
        outport.send(note_off(root +a))
        outport.send(note_off(root + b))
        outport.send(note_off(root-12+b))
        outport.send(note_off(root - 12 + a))

def strum(root,chord_type,bpm,is_natural_sounding,strum):

    if strum==strum_pattern.DDUUDD:

        duration=4/16*60/bpm

        Chord(root, duration*4,is_natural_sounding,True,chord_type)
        Chord(root, duration*3, is_natural_sounding,True,chord_type)
        Chord(root, duration*2, is_natural_sounding,False,chord_type)
        Chord(root, duration, is_natural_sounding,False,chord_type)
        Chord(root, duration*2, is_natural_sounding,True,chord_type)
        Chord(root, duration*2, is_natural_sounding,True,chord_type)
        Chord(root, duration , is_natural_sounding, True,chord_type)
        Chord(root, duration, is_natural_sounding, False,chord_type)

    elif strum==strum_pattern.DDUUDU:

        duration=2/12*60/bpm
        Chord(root, duration * 3, is_natural_sounding, True,chord_type)
        Chord(root, duration*2, is_natural_sounding, True,chord_type)
        Chord(root, duration*3, is_natural_sounding, False,chord_type)
        Chord(root, duration, is_natural_sounding, False,chord_type)
        Chord(root, duration*2, is_natural_sounding, True,chord_type)
        Chord(root, duration, is_natural_sounding, True,chord_type)

    elif strum == strum_pattern.DDU:

        duration = 1 / 6 * 60 / bpm
        Chord(root, duration * 3, is_natural_sounding, True,chord_type)
        Chord(root, duration*2, is_natural_sounding, True,chord_type)
        Chord(root, duration, is_natural_sounding, False,chord_type)
    elif strum == strum_pattern.DDUDUDUDUDU:
        
        duration = 4 / 17.5 * 60 / bpm
        Chord(root, duration *3.5, is_natural_sounding, True,chord_type)
        Chord(root, duration, is_natural_sounding, True,chord_type)
        Chord(root, duration, is_natural_sounding, False,chord_type)
        Chord(root, duration, is_natural_sounding, True,chord_type)
        Chord(root, duration*3.5, is_natural_sounding, False,chord_type)
        Chord(root, duration, is_natural_sounding, True,chord_type)
        Chord(root, duration, is_natural_sounding, False,chord_type)
        Chord(root, duration, is_natural_sounding, True,chord_type)
        Chord(root, duration*2.5 , is_natural_sounding, False,chord_type)
        Chord(root, duration, is_natural_sounding, True,chord_type)
        Chord(root, duration, is_natural_sounding, True, chord_type)
    elif strum==strum_pattern.DDDD:
        
        duration = 1 / 2 * 60 / bpm
        Chord(root, duration , is_natural_sounding, True, chord_type)
        Chord(root, duration, is_natural_sounding, True, chord_type)








##################################################Automatic music creator#################################





def chords_progression():
    
    d = {0: [C, chord_type.Major], 1 :[D, chord_type.Minor], 2 : [E,  chord_type.Minor],
         3 : [F, chord_type.Major], 4 :[G, chord_type.Major], 5 :[A,  chord_type.Minor]}
    P = np.array([[0.25, 0.05,0.05, 0.25,0.25,0.15],
              [0, 0.1, 0,0,0.9,0],
              [0.1, 0, 0.2,0,0,0.7],
              [0.3, 0, 0,0.2,0.3,0.2],
              [0.3, 0, 0,0.3,0.3,0.1],
              [0.1, 0.8, 0,0,0,0.1]])
    state=np.zeros(6) 
    state[random.randint(0,6)] = 1
    chords = [d[np.argmax(state)]]
    for x in range(4):
        proba = np.dot(state,P)
        state=np.zeros(6) 
        state[int(np.random.choice(6, 1, p=proba))] = 1
        chords.append(d[np.argmax(state)])
        
    return chords


#####################################################emotion detection + music creation#####################


tokens = tokenizer.encode(data.loc[1,"poems"], return_tensors='pt')[0:,:512]
result = model(tokens)
result.logits
score = int(torch.argmax(result.logits))+1


if score < 3:

    bpm = 35
    strum(C, chord_type.Minor, bpm, True, strum_pattern.DDDD)
    strum(F, chord_type.Minor, bpm, True, strum_pattern.DDDD)
    strum(D, chord_type.Diminished, bpm, True, strum_pattern.DDDD)
    strum(G, chord_type.Major, bpm, True, strum_pattern.DDDD)

else: 
    bpm = 60
    chords = chords_progression()
    strum(chords[0][0], chords[0][1], bpm, True, strum_pattern.DDUUDU)
    strum(chords[1][0], chords[1][1], bpm, True, strum_pattern.DDUUDU)
    strum(chords[2][0], chords[2][1], bpm, True, strum_pattern.DDUUDU)
    strum(chords[3][0], chords[3][1], bpm, True, strum_pattern.DDUUDU)
    strum(chords[4][0], chords[4][1], bpm, True, strum_pattern.DDUUDU)
