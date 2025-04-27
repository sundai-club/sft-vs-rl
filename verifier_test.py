import countdown_verifier

llm_response = "Assistant: To create an equation that equals 98 using the numbers 44, 19, and 35, let's consider the operations and how we can combine these numbers:\n\n1. We need to get close to 98 using the given numbers.\n2. Multiplication can quickly increase the value, so I'll try to use it.\n3. Let's see if 44 can be used in multiplication with 19 or 35:\n   - 4419 is too high.\n   - 44*35 is also too high.\n4. We can try adding or subtracting after multiplying a number with another.\n5. Multiplying 35 with 19 gives 665, which is much higher than 98, so we should avoid using multiplication for these two.\n6. We need to use addition or subtraction to get close to 98. Since 44 is close to half of 98, we can use this.\n7. We need to find the difference between 98 and 44, which is 54.\n8. 54 can be achieved by adding 19 and 35.\n\nSo, the equation can be: 44 + (19 + 35) = 98.\n</think>\n<answer> 44 + (19 + 35) </answer>"

ground_truth = {
    'target': 98,
    'numbers': [44, 19, 35]
}

score = countdown_verifier.compute_score(
    solution_str=llm_response,
    ground_truth=ground_truth,
    method='strict',
    format_score=0.1,
    score=1.
)

print(f"Score: {score}")