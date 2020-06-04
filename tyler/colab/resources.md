## Resources
I used the following blog post for most of the padding/packing code.

https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e


However, the solution above had a strange (IMO) approach to computing the loss. I just wanted to get the last LSTM output from each sentence. I used the code below for this.

https://stackoverflow.com/questions/55399115/get-each-sequences-last-item-from-packed-sequence