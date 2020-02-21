def read_lines_into_list(filename):
    content = []
    with open(filename) as f:
        for line in f:
            content.append(line.strip())
    # # you may also want to remove whitespace characters like `\n` at the end of each line
    # content = [x.strip() for x in content]
    return content
