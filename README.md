# IMDB---Analise-Classificatoria-ML
Projeto de machine learning com python/jupyter notebook 


```python
# Este notebbok tem o objetivo de fazer um modelo de machine learning,
# que seja capaz de realizar uma previsão classificatoria binaria do datasheet sobre reviews de filmes do portal IMDB
# contido neste diretorio
```


```python
import numpy as np 
import pandas as pd 
import re
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
import pickle
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\Giovanni\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\Giovanni\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    


```python
data = pd.read_csv('IMDB-Dataset.csv')
print(data.shape)
data.head()
```

    (50000, 2)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>One of the other reviewers has mentioned that ...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A wonderful little production. &lt;br /&gt;&lt;br /&gt;The...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I thought this was a wonderful way to spend ti...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Basically there's a family where a little boy ...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Petter Mattei's "Love in the Time of Money" is...</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 50000 entries, 0 to 49999
    Data columns (total 2 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   review     50000 non-null  object
     1   sentiment  50000 non-null  object
    dtypes: object(2)
    memory usage: 781.4+ KB
    


```python
data.sentiment.value_counts()
```




    negative    25000
    positive    25000
    Name: sentiment, dtype: int64




```python
data.sentiment.replace('positive',1,inplace=True)
data.sentiment.replace('negative',0,inplace=True)
data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>One of the other reviewers has mentioned that ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A wonderful little production. &lt;br /&gt;&lt;br /&gt;The...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I thought this was a wonderful way to spend ti...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Basically there's a family where a little boy ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Petter Mattei's "Love in the Time of Money" is...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Probably my all-time favorite movie, a story o...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>I sure would like to see a resurrection of a u...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>This show was an amazing, fresh &amp; innovative i...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Encouraged by the positive comments about this...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>If you like original gut wrenching laughter yo...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.review[0]
```




    "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side."




```python
#Limpando as tags HTML das reviews
def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',text)
data.review = data.review.apply(clean)
data.review[0]
```




    "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side."




```python
#Removendo caracteres especiais
def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem
data.review = data.review.apply(is_special)
data.review[0]
```




    'One of the other reviewers has mentioned that after watching just 1 Oz episode you ll be hooked  They are right  as this is exactly what happened with me The first thing that struck me about Oz was its brutality and unflinching scenes of violence  which set in right from the word GO  Trust me  this is not a show for the faint hearted or timid  This show pulls no punches with regards to drugs  sex or violence  Its is hardcore  in the classic use of the word It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary  It focuses mainly on Emerald City  an experimental section of the prison where all the cells have glass fronts and face inwards  so privacy is not high on the agenda  Em City is home to many  Aryans  Muslims  gangstas  Latinos  Christians  Italians  Irish and more    so scuffles  death stares  dodgy dealings and shady agreements are never far away I would say the main appeal of the show is due to the fact that it goes where other shows wouldn t dare  Forget pretty pictures painted for mainstream audiences  forget charm  forget romance   OZ doesn t mess around  The first episode I ever saw struck me as so nasty it was surreal  I couldn t say I was ready for it  but as I watched more  I developed a taste for Oz  and got accustomed to the high levels of graphic violence  Not just violence  but injustice  crooked guards who ll be sold out for a nickel  inmates who ll kill on order and get away with it  well mannered  middle class inmates being turned into prison bitches due to their lack of street skills or prison experience  Watching Oz  you may become comfortable with what is uncomfortable viewing    thats if you can get in touch with your darker side '




```python
#Transformando em letras minusculas

def to_lower(text):
    return text.lower()
data.review = data.review.apply(to_lower)
data.review[0]
```




    'one of the other reviewers has mentioned that after watching just 1 oz episode you ll be hooked  they are right  as this is exactly what happened with me the first thing that struck me about oz was its brutality and unflinching scenes of violence  which set in right from the word go  trust me  this is not a show for the faint hearted or timid  this show pulls no punches with regards to drugs  sex or violence  its is hardcore  in the classic use of the word it is called oz as that is the nickname given to the oswald maximum security state penitentary  it focuses mainly on emerald city  an experimental section of the prison where all the cells have glass fronts and face inwards  so privacy is not high on the agenda  em city is home to many  aryans  muslims  gangstas  latinos  christians  italians  irish and more    so scuffles  death stares  dodgy dealings and shady agreements are never far away i would say the main appeal of the show is due to the fact that it goes where other shows wouldn t dare  forget pretty pictures painted for mainstream audiences  forget charm  forget romance   oz doesn t mess around  the first episode i ever saw struck me as so nasty it was surreal  i couldn t say i was ready for it  but as i watched more  i developed a taste for oz  and got accustomed to the high levels of graphic violence  not just violence  but injustice  crooked guards who ll be sold out for a nickel  inmates who ll kill on order and get away with it  well mannered  middle class inmates being turned into prison bitches due to their lack of street skills or prison experience  watching oz  you may become comfortable with what is uncomfortable viewing    thats if you can get in touch with your darker side '




```python
#Removendo stopwords

def rem_sw(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

data.review = data.review.apply(rem_sw)
data.review[0]
```




    ['one',
     'reviewers',
     'mentioned',
     'watching',
     '1',
     'oz',
     'episode',
     'hooked',
     'right',
     'exactly',
     'happened',
     'first',
     'thing',
     'struck',
     'oz',
     'brutality',
     'unflinching',
     'scenes',
     'violence',
     'set',
     'right',
     'word',
     'go',
     'trust',
     'show',
     'faint',
     'hearted',
     'timid',
     'show',
     'pulls',
     'punches',
     'regards',
     'drugs',
     'sex',
     'violence',
     'hardcore',
     'classic',
     'use',
     'word',
     'called',
     'oz',
     'nickname',
     'given',
     'oswald',
     'maximum',
     'security',
     'state',
     'penitentary',
     'focuses',
     'mainly',
     'emerald',
     'city',
     'experimental',
     'section',
     'prison',
     'cells',
     'glass',
     'fronts',
     'face',
     'inwards',
     'privacy',
     'high',
     'agenda',
     'em',
     'city',
     'home',
     'many',
     'aryans',
     'muslims',
     'gangstas',
     'latinos',
     'christians',
     'italians',
     'irish',
     'scuffles',
     'death',
     'stares',
     'dodgy',
     'dealings',
     'shady',
     'agreements',
     'never',
     'far',
     'away',
     'would',
     'say',
     'main',
     'appeal',
     'show',
     'due',
     'fact',
     'goes',
     'shows',
     'dare',
     'forget',
     'pretty',
     'pictures',
     'painted',
     'mainstream',
     'audiences',
     'forget',
     'charm',
     'forget',
     'romance',
     'oz',
     'mess',
     'around',
     'first',
     'episode',
     'ever',
     'saw',
     'struck',
     'nasty',
     'surreal',
     'say',
     'ready',
     'watched',
     'developed',
     'taste',
     'oz',
     'got',
     'accustomed',
     'high',
     'levels',
     'graphic',
     'violence',
     'violence',
     'injustice',
     'crooked',
     'guards',
     'sold',
     'nickel',
     'inmates',
     'kill',
     'order',
     'get',
     'away',
     'well',
     'mannered',
     'middle',
     'class',
     'inmates',
     'turned',
     'prison',
     'bitches',
     'due',
     'lack',
     'street',
     'skills',
     'prison',
     'experience',
     'watching',
     'oz',
     'may',
     'become',
     'comfortable',
     'uncomfortable',
     'viewing',
     'thats',
     'get',
     'touch',
     'darker',
     'side']




```python
def stem(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

data.review = data.review.apply(stem)
data.review[0]
```




    'one review mention watch 1 oz episod hook right exact happen first thing struck oz brutal unflinch scene violenc set right word go trust show faint heart timid show pull punch regard drug sex violenc hardcor classic use word call oz nicknam given oswald maximum secur state penitentari focus main emerald citi experiment section prison cell glass front face inward privaci high agenda em citi home mani aryan muslim gangsta latino christian italian irish scuffl death stare dodgi deal shadi agreement never far away would say main appeal show due fact goe show dare forget pretti pictur paint mainstream audienc forget charm forget romanc oz mess around first episod ever saw struck nasti surreal say readi watch develop tast oz got accustom high level graphic violenc violenc injustic crook guard sold nickel inmat kill order get away well manner middl class inmat turn prison bitch due lack street skill prison experi watch oz may becom comfort uncomfort view that get touch darker side'




```python
#Criando o modelo
```


```python
#Criando bag of words

x = np.array(data.iloc[:,0].values)
y = np.array(data.sentiment.values)
cv = CountVectorizer(max_features=1000)
x = cv.fit_transform(data.review).toarray()
print("x = ", x.shape)
print("y = ", y.shape)
```

    x =  (50000, 1000)
    y =  (50000,)
    


```python
#Test split

trainx,testx,trainy,testy = train_test_split(x,y,test_size=0.2,random_state=9)
print("Treino: x = {}, y = {}".format(trainx.shape, trainy.shape))
print("Teste: x = {}, y = {} ".format(testx.shape, testy.shape))
      
```

    Treino: x = (40000, 1000), y = (40000,)
    Teste: x = (10000, 1000), y = (10000,) 
    


```python
#Definindo os modelos e treinando eles

gnb,mnb,bnb = GaussianNB(),MultinomialNB(alpha=1.0,fit_prior=True),BernoulliNB(alpha=1.0,fit_prior=True)
gnb.fit(trainx, trainy)
mnb.fit(trainx, trainy)
bnb.fit(trainx, trainy)
```




    BernoulliNB()




```python
#Decidindo entre prediction e accuracy

ypg = gnb.predict(testx)
ypm = mnb.predict(testx)
ypb = bnb.predict(testx)

print("Gaussian = ",accuracy_score(testy,ypg))
print("Multinomial = ",accuracy_score(testy,ypm))
print("Bernoulli = ",accuracy_score(testy,ypb))
```

    Gaussian =  0.7843
    Multinomial =  0.831
    Bernoulli =  0.8386
    


```python
pickle.dump(bnb,open('model1.pkl','wb'))
rev =  """Terrible. Complete trash. Brainless tripe. Insulting to anyone who isn't an 8 year old fan boy. Im actually pretty disgusted that this movie is making the money it is - what does it say about the people who brainlessly hand over the hard earned cash to be 'entertained' in this fashion and then come here to leave a positive 8.8 review?? Oh yes, they are morons. Its the only sensible conclusion to draw. How anyone can rate this movie amongst the pantheon of great titles is beyond me.

So trying to find something constructive to say about this title is hard...I enjoyed Iron Man? Tony Stark is an inspirational character in his own movies but here he is a pale shadow of that...About the only 'hook' this movie had into me was wondering when and if Iron Man would knock Captain America out...Oh how I wished he had :( What were these other characters anyways? Useless, bickering idiots who really couldn't organise happy times in a brewery. The film was a chaotic mish mash of action elements and failed 'set pieces'...

I found the villain to be quite amusing.

And now I give up. This movie is not robbing any more of my time but I felt I ought to contribute to restoring the obvious fake rating and reviews this movie has been getting on IMDb."""
f1 = clean(rev)
f2 = is_special(f1)
f3 = to_lower(f2)
f4 = rem_sw(f3)
f5 = stem(f4)

bow,words = [], word_tokenize(f5)
for word in words:
    bow.append(words.count(word))
    
word_dict = cv.vocabulary_
pickle.dump(word_dict, open('bow.pkl','wb'))

```


```python
inp = []
for i in word_dict:
    inp.append(f5.count(i[0]))
y_pred = bnb.predict(np.array(inp).reshape((1, 1000)))
print(y_pred)
```

    [0]
    


```python
#Resultado [0] = Negativo
#A previsão da review (rev) deu negativa como esperado
```

