from glob import glob
import random

include_self_anchor = False

def write_file(filename, people):
    with open(filename, 'w') as t:
        neg_person = people[-1]
        t.write('id,anchor_img,pos_img,neg_img\n')
        c = 1
        for person in people:
            wrote = False
            neg_name = neg_person.rsplit('/',1)[-1]
            neg_img = neg_person + '/' + neg_name +'_0001.jpg'
            name = person.rsplit('/',1)[-1]
            anchor_img = person + '/' + name +'_0001.jpg'
            for pos_img in glob(person+'/*'):
                if include_self_anchor or anchor_img != pos_img:
                    t.write(str(c)+','+anchor_img+','+pos_img+','+neg_img+'\n')
                    wrote = True
            neg_person = person
            if wrote:
                c += 1

people = "./lfw/*"
people = glob(people)
random.shuffle(people)
split_size = int(len(people) * .10)
train_set = people[0:(1-split_size)]
val_set = people[(1-split_size):len(people)]


write_file('train1.csv', train_set)
write_file('val1.csv', val_set)

