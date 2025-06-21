from math import sqrt

def read_file(file_path: str):
  with open(file_path, encoding = "utf-8") as file:
    return sorted(map(float, file.read().split()))

def mean(values: list[float]):
  if not values: return 0.0
  return sum(values) / len(values)

def mode(values: list[float]):
  if not values: return 0.0
  frequency: dict[float, int] = {}
  for value in values:
    try: frequency[value] += 1
    except KeyError: frequency[value] = 1
  key, v = max(frequency.items(), key = lambda x: x[1])
  return key

def median(values: list[float]):
  if not values: return 0.0
  div, mod = divmod(len(values), 2)
  temp = values[div + mod - 1]
  return (temp + values[div]) / 2

def quartis(values: list[float]):
  if not values: return 0.0, 0.0, 0.0
  div, mod = divmod(len(values), 2)
  q1 = median(values[:div])
  temp = values[div + mod - 1]
  q2 = (temp + values[div]) / 2
  q3 = median(values[div + mod:])
  return q1, q2, q3

def amplitude(values: list[float]):
  if not values: return 0.0
  return max(values) - min(values)

def variance(values: list[float]):
  if not values: return 0.0
  mean_value = mean(values)
  sums = sum((value - mean_value) ** 2 for value in values)
  return sums / len(values)

def deviation(values: list[float]):
  if not values: return 0.0
  return sqrt(variance(values))

def coef_variation(values: list[float]):
  if not values: return 0.0
  mean_value = mean(values)
  if not mean_value: return 0.0
  return 100 * deviation(values) / mean_value

def outliers(values: list[float]):
  if not values: return values
  q1, q2, q3 = quartis(values)
  tree_halfs_iqr = 1.5 * (q3 - q1)
  lower, upper = q1 - tree_halfs_iqr, q3 + tree_halfs_iqr
  return [value for value in values if value < lower or value > upper]

def remove_outliers(values: list[float]):
  if not values: return values
  q1, q2, q3 = quartis(values)
  tree_halfs_iqr = 1.5 * (q3 - q1)
  lower, upper = q1 - tree_halfs_iqr, q3 + tree_halfs_iqr
  return [value for value in values if lower <= value <= upper]

def main():
  file_path = "./Lista 00/entrada.txt"

  print("Respostas com outliers no conjunto:")
  values = read_file(file_path)
  mean_values = mean(values)
  print(f"A média dos valores é: {mean_values}")
  mode_value = mode(values)
  print(f"A moda dos valores é: {mode_value}")
  median_value = median(values)
  print(f"A mediana dos valores é: {median_value}")
  q1, q2, q3 = quartis(values)
  print(f"Q1: {q1}, Q2: {q2}, Q3: {q3}")
  amplitude_value = amplitude(values)
  print(f"A amplitude dos valores é: {amplitude_value}")
  variance_value = variance(values)
  print(f"A variância dos valores é: {variance_value}")
  deviation_value = deviation(values)
  print(f"O desvio padrão dos valores é: {deviation_value}")
  coef_var_value = coef_variation(values)
  print(f"O coef. de variação dos valores é: {coef_var_value}%")
  outliers_values = outliers(values)
  print(f"Valores considerados outliers: {outliers_values}")

  print("\n" + "=" * 50 + "\n")
  print("Respostas sem outliers no conjunto:")
  values = remove_outliers(values)
  mean_values = mean(values)
  print(f"A média dos valores é: {mean_values}")
  mode_value = mode(values)
  print(f"A moda dos valores é: {mode_value}")
  median_value = median(values)
  print(f"A mediana dos valores é: {median_value}")
  q1, q2, q3 = quartis(values)
  print(f"Q1: {q1}, Q2: {q2}, Q3: {q3}")
  amplitude_value = amplitude(values)
  print(f"A amplitude dos valores é: {amplitude_value}")
  variance_value = variance(values)
  print(f"A variância dos valores é: {variance_value}")
  deviation_value = deviation(values)
  print(f"O desvio padrão dos valores é: {deviation_value}")
  coef_var_value = coef_variation(values)
  print(f"O coef. de variação dos valores é: {coef_var_value}%")

if __name__ == "__main__": main()