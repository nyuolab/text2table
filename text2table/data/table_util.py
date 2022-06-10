# takes a sequence representation of a table and converts it to a csv
def seq_to_csv(seq):
    csv = seq.replace(" <COL> ", ",").replace(" <ROW> ", "\n")
    return csv

# takes a list of csv strings and concats them
# assumes the csv strings in the list only have 2 rows and the first row is the header row
# assumes all csv strings have the same headers
def concat_table(csv_list):
    # gets the header of the table
    table = csv_list[0].split("\n")[0] + "\n"
    # adds the nonheader row in each csv string to the complete table
    for csv in csv_list:
        table = table + csv.split("\n")[1] + "\n"
    return table.strip()

