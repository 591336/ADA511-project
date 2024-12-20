import random

def hitsvsgain(ntrials, chooseAtrueA, chooseAtrueB, chooseBtrueB, chooseBtrueA, probsA=[0.5]):
    ## Recycle & shuffle the given probabilities for the number of trials
    probsArepeated = random.choices(probsA, k=ntrials)
    
    ## "Magic" parameters used in making the optimal decision
    threshold1 = (chooseAtrueA - chooseAtrueB + chooseBtrueB - chooseBtrueA)
    threshold2 = (chooseBtrueB - chooseAtrueB)
    
    ## Initialize total "hits" and gains
    ## 'mlc' refers to the Machine-Learning Classifier
    ## 'opm' refers to the Optimal Predictor Machine
    mlchits = 0
    mlcgain = 0
    opmhits = 0
    opmgain = 0
    
    ##
    ## Loop through the trials and their probabilities
    for probabilityA in probsArepeated:
        ## Output of the MLC, based on the current probability
        if probabilityA > 0.5:
            mlcchoice = 'A'
        elif probabilityA < 0.5:
            mlcchoice = 'B'
        else:
            mlcchoice = random.choice(['A', 'B']) # A or B with 50%/50% prob.
        
        ## Output of the OPM, based on the current probability
        if threshold1 * probabilityA > threshold2:
            opmchoice = 'A'
        elif threshold1 * probabilityA < threshold2:
            opmchoice = 'B'
        else:
            opmchoice = random.choice(['A', 'B']) # A or B with 50%/50% prob.
        
        ##
        ## Correct answer for the current trial
        trueitem = random.choices(['A', 'B'], weights=[probabilityA, 1-probabilityA], k=1)[0]
        
        ##
        ## MLC: add one "hit" if correct guess, and add gain/loss
        if mlcchoice == trueitem:
            mlchits += 1 # one success
            if trueitem == 'A':
                mlcgain += chooseAtrueA
            else:
                mlcgain += chooseBtrueB
        else:
            if trueitem == 'B':
                mlcgain += chooseAtrueB
            else:
                mlcgain += chooseBtrueA
        
        ##
        ## OPM: add one "hit" if correct guess, and add gain/loss
        if opmchoice == trueitem:
            opmhits += 1 # one success
            if trueitem == 'A':
                opmgain += chooseAtrueA
            else:
                opmgain += chooseBtrueB
        else:
            if trueitem == 'B':
                opmgain += chooseAtrueB
            else:
                opmgain += chooseBtrueA
    
    ## end of loop
    ##
    ## Output total number of hits and total gain or loss produced
    print('\nTrials:', ntrials)
    print('Machine-Learning Classifier: successes', mlchits, '(', format(mlchits/ntrials*100, '.3f'), '%)',
          '| total gain', mlcgain)
    print('Optimal Predictor Machine:   successes', opmhits, '(', format(opmhits/ntrials*100, '.3f'), '%)',
          '| total gain', opmgain)
    print('\n')