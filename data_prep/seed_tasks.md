# TB Seed Tasks for Instruction Tuning
# 25 human-written seed examples for generating synthetic TB instruction-response pairs
You are asked to come up with a set of 20 diverse task instructions specifically related to the Mycobacterium tuberculosis. These task instructions will be given to a GPT model and I will evaluate the GPT model for completing the instructions and fine-tune it accordingly.
Here are the requirements:
1. Try to ensure a diverse set of tasks by varying the actions required in each instruction.
2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instructions.
3. The type of instructions should be diverse. The list should include diverse types of tasks like question answering, classification, summarization, simplification, hypothesis generation, reasoning etc.
4. A GPT language model should be able to complete the instruction. For example, do not ask the assistant to create any visual or audio output.
5. The instructions should be in English.
6. The instructions should be concise and comprehensive. Either an imperative sentence or a question is permitted.
7. You should generate an appropriate input to the instruction. The input field should contain a specific example provided for the instruction. It should involve realistic data and should not contain simple placeholders. The input should provide substantial content to make the instruction challenging.
8. The output should be an appropriate response to the instruction and the input.

###
Instruction: Given a detailed health-related query from a consumer, summarize it into a concise question.
Input: Hi, I have mixed diagnosis of autoimmune connective tissue disease and would like to submit information about this
disease. I also need some serious rheumatologist recommended in Argentina or Latin America. From already thank you very
much.
Output: How can I find rheumatologist(s) who specialize in mixed connective tissue disease in Latin America?
Instruction: Given a section of a medical record and a question related to it, provide the relevant information from the record as
an answer.
Input: 08/31/96 ascending aortic root replacement with homograft with omentopexy. The patient continued to be hemodynami-
cally stable making good progress. Physical examination: BMI: 33.4 Obese, high risk. Pulse: 60. resp. rate: 18. Has the patient
ever had an abnormal BMI?
Output: BMI: 33.4 Obese, high risk
Instruction: Given a pair of sentences, determine if the second sentence is an entailment, contradiction, or neutral with respect
to the first one.
Input: Sentence1: Labs were notable for Cr 1.7 (baseline 0.5 per old records) and lactate 2.4.
Sentence2: Patient has elevated Cr.
Output: entailment
Instruction: Given a medical scenario and a multiple choice question, choose the correct option.
Input: A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination. She states it started 1 day ago
and has been worsening despite drinking more water and taking cranberry extract. She otherwise feels well and is followed by a
doctor for her pregnancy. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are
19/min, and oxygen saturation is 98% on room air. Physical exam is notable for an absence of costovertebral angle tenderness
and a gravid uterus. Which of the following is the best treatment for this patient? Options: A) Ampicillin B) Ceftriaxone C)
Ciprofloxacin D) Doxycycline E) Nitrofurantoin
Output: E) Nitrofurantoin