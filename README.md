## Requirement

- Python2.7
- Pytorch 1.4.0
- numpy
- nltk
- nlgeval [https://github.com/Maluuba/nlg-eval]

# Train the model.
python main.py

# Evaluate the model.
Metrics ## - BLEU - METEOR - ROUGE - CIDEr

python eval_BLEU.py --test MWPGen


#Metircs 
Solvability of the generated math word problems. Evaluated by a state-of-art math word problem solving model GTS [https://github.com/ShichaoSun/math_seq2tree]

python eval_NLG.py --test MWPGen
