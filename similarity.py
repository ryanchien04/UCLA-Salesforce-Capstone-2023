from Levenshtein import distance

assert(distance('kitten', 'sitting') == 3)

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