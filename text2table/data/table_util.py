# takes a sequence representation of a table and converts it to a csv
def seq_to_csv(seq):
    csv = seq.replace(" <COL> ", ",").replace(" <ROW> ", "\n")
    return csv

# takes a list of csv strings and concats them
def concat_table(csv_list):
    table = csv_list[0].split("\n")[0] + "\n"
    for csv in csv_list:
        table = table + csv.split("\n")[1] + "\n"
    return table.strip()

