from glob import glob

include_self_anchor = False

people = "./lfw/*"
neg_person = glob(people)[-1]
with open('./train1.csv', 'w') as t:
    t.write('id,anchor_img,pos_img\n')
    c = 1
    for person in glob(people):
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
