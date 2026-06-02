# Experiment with curriculum learning to see if we can do OOD better

## Lesson 1: Reversal to ANS for upto 22 digits
  * Input data will be random numbers (positive and negative numbers) of length upto 22 digits prefixed by [REV] and suffixed by [ANS].
  * Output will be the reversed digits followed by [EOS].

  Example: [REV]123456[ANS]654321[EOS]
  Example: [REV]-123456[ANS]-654321[EOS]

## Lesson 2: Reversal of multiple numbers
  * Input data will be 1 to 6 random positve and negative numbers of length upto 9 digits separated by spaces prefixed by [BOS] and suffixed by [REV].
  * 2nd number onwards will have an operator (+ or -) after the number (RPN notation)
  * Output after [REV] will be the reversed digits of all numbers separated by [SEP] suffixed by [MATH].
  * operators will stay the same place after reversal.

  Example: [BOS]123 456+[REV]321[SEP]654+[MATH]
  Example: [BOS]-123 456- 234+[REV]-321[SEP]654-[SEP]432+[MATH]
  
## Lesson 3: Math operation afer reversal of numbers
 * Input data will be 1 to 6 random positve and negative numbers of length upto 9 digits separated by [SEP] (not spaces) and prefexed by [REV].
 * 2nd number onwards will have an operator (+ or -) after the number (RPN notation).
 * Input ends with [MATH]
 * Note: This input is exactly same as output of Lesson 2.
 * Output will solution of digit math as we have been doing in create_dataset.py
 * Note: As MATH only operates on 1st two numbers after [REV] if more numbers are present they will be copied after the result of first two numbers. (including [SEP]. This will ensure [MATH] has all the data needed for next [REV] phase.)
 * end output with [REV]

 ## Lesson 4: answer number and recreate REV phase for recursion.
  * Input data will be output of Lesson 3 (all MATH) prefixed by [MATH] and suffixed by [REV] 
  * output will be the number produced by the MATH operation followed by [SEP] if [SEP] is in [MATH] phase, then end with [MATH]
  * If there is no [SEP] in [MATH]  that means we are done so end with [ANS]
