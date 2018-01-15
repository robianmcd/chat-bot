delim = ' +++$+++ '

lines = {}
conversations = []

with open('movie_lines.txt', 'r', encoding='utf-8') as lines_file:
    for row in lines_file:
            parts = row.split(delim)
            lines[parts[0]] = parts[4].replace("\n", '')

with open('movie_conversations.txt', 'r', encoding='utf-8') as conversations_file:
    for row in conversations_file:
        parts = row.split(delim)
        conv_lines = parts[3]\
            .replace('[', '')\
            .replace(']', '')\
            .replace("'", '')\
            .replace("\n", '')\
            .split(', ')

        conversations.append([lines[line] for line in conv_lines])

with open('data_responses.txt', 'w', encoding='utf-8') as responses_file:
    with open('data_context.txt', 'w', encoding='utf-8') as context_file:
        for conversation in conversations:
            for i in range(1, len(conversation)):
                if i is 1:
                    first_line = ''
                else:
                    first_line = conversation[i-2] + ' '
                second_line = conversation[i-1]
                third_line = conversation[i]
                context_file.write(first_line + second_line + '\n')
                responses_file.write(third_line + '\n')