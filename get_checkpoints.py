def process_file(file_path):
    results = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for i in range(1, len(lines)):
        if "Saving model and optimizer state at iteration" in lines[i]:
            previous_line = lines[i - 1]
            if 'Saving model' in previous_line:
                continue

            start = previous_line.find('[') + 1
            end = previous_line.find(']')
            if start == 0 or end == -1:
                continue

            numbers = previous_line[start:end].split(',')
            numbers = [float(num.strip()) for num in numbers]

            if len(numbers) >= 7:
                # num_sum = sum(numbers[1:6])
                num_sum = numbers[1]
                id_num = numbers[7]
                results.append((id_num, num_sum))

    results.sort(key=lambda x: x[1])

    return results

file_path = './logs/csmsc/train.log'
sorted_results = process_file(file_path)
for id_num, num_sum in sorted_results:
    print(f"ID: {id_num}, Sum: {num_sum}")
