labels = ['.',  'K','Q','B','N','R','P'  ,'k','q','b','n','r','p']
# print(len(labels))
labels_dict = {labels[i] : i for i in range(len(labels))}
# print(labels_dict)

columns_to_encode = ['p'+str(i) for i in range(64)]
test = data_pandas[columns_to_encode].iloc[0].to_list()

def one_hot_encode_labels(labels, labels_dict):
    encoded_labels = torch.zeros(len(labels), len(labels_dict))
    for i, label in enumerate(labels):
        encoded_labels[i][labels_dict[label]] = 1
    return encoded_labels

# print(labels_dict)
# for i in range(64):
#     print(test[i],t[i])


t = one_hot_encode_labels(test, labels_dict)

t = t.view(-1, 64*13)

p = torch.Tensor([0])
# print(p)
# print(t)

print(t.shape)
print(p.shape)

appended_tensor = torch.cat((t, p.unsqueeze(1)), dim=1)

print(appended_tensor.shape)


# Origin file
o_file = ['a','b','c','d','e','f','g','h']

# Origin rank
o_rank = [1,2,3,4,5,6,7,8]

# Piece neurons (might not be needed since the info is already there in the origin to destination)
# piece = ['K','Q','B','N','R','P'] 

# Destination file
d_file = ['a','b','c','d','e','f','g','h']

# Destination rank
d_rank = [1,2,3,4,5,6,7,8]

# Edge cases
promotion_type = ['q','r','b','n']

# output = o_rank+o_file+d_rank+d_file+promotion_type
# print(len(output))
# output_labels = promotion_type
# print(len(output_labels))
# output_labels_dict = {output_labels[i] : i for i in range(len(output_labels))}
# print(output_labels_dict)


# print(test2)






# Origin file
o_file = ['a','b','c','d','e','f','g','h']

# Origin rank
o_rank = [1,2,3,4,5,6,7,8]

# Piece neurons (might not be needed since the info is already there in the origin to destination)
# piece = ['K','Q','B','N','R','P'] 

# Destination file
d_file = ['a','b','c','d','e','f','g','h']

# Destination rank
d_rank = [1,2,3,4,5,6,7,8]

# Edge cases
promotion_type = ['q','r','b','n']

# output = piece+o_rank+o_file+d_rank+d_file+promotion_type+castling
output_labels = o_file+o_rank+d_file+d_rank+promotion_type
print(len(output_labels))
output_labels_dict = {output_labels[i] : i for i in range(len(output_labels))}
print(output_labels_dict)


test2 = data_pandas['uci']
print(test2)






# def decode_uci(tensor : torch.Tensor):
#     tensor_parts = torch.split(tensor, [8, 8, 8, 8, 4], dim=1)

#     # Print the shapes of the split tensors
#     for part in tensor_parts:
#         print(part)

# decode_uci(t)


# Testing
position_columns = ['p'+str(i) for i in range(64)]
test_position = data_pandas[position_columns].iloc[0].to_list()
test_player = data_pandas['player'].iloc[0]

print('input:', test_player,test_position)

model = Neuro_gambit()
forward_test = model.forward_pandas(test_position, test_player)

print('output:', forward_test)




# Testing 
# test2 = data_pandas['uci'].iloc[418]
test2 = data_pandas['uci'].iloc[0]
print(test2)
t = encode_uci(test2)
t_split = torch.split(t, [8, 8, 8, 8, 4], dim=1)

print(decode_uci(t_split))