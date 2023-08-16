from Levenshtein import distance

assert(distance('kitten', 'sitting') == 3)
assert(distance('sitting', 'sitting') == 0)

def exact_match_accuracy(reference, hypothesis):
  """Calculates the exact match accuracy between two sequences.

  Args:
    reference: The reference sequence.
    hypothesis: The hypothesis sequence.

  Returns:
    The exact match accuracy.
  """

  # Calculate the number of matches.
  matches = 0
  for i in range(len(reference) - len(hypothesis) + 1):
    if reference[i:i + len(hypothesis)] == hypothesis:
      matches += 1

  # Calculate the precision.
  precision = matches / (matches + (len(reference) - len(hypothesis)) / len(hypothesis))

  return precision

assert(exact_match_accuracy('kitten', 'sitting') == 0)
assert(exact_match_accuracy('sitting', 'sitting') == 1)
print(exact_match_accuracy('sitting on a tree', 'sitting'))
