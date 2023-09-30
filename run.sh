#!/bin/bash

mkdir -p compiled images

rm -f ./compiled/*.fst ./images/*.pdf

# ############ Compile source transducers ############
for transducer in sources/*.txt tests/*.txt; do
    echo "Compiling: $transducer"
    fstcompile --isymbols=syms.txt --osymbols=syms.txt "$transducer" | fstarcsort > "compiled/$(basename "$transducer" .txt).fst"
done


# ############ CORE OF THE PROJECT  ############

# create mix2numerical.fst transducer
fstconcat compiled/mmm2mm.fst compiled/aux_datareader.fst > compiled/mix2numerical.fst

# create pt2en.fst transducer
fstconcat compiled/aux_pt_en_translate.fst compiled/aux_datareader.fst > compiled/pt2en.fst

# create en2pt.fst transducer
fstinvert compiled/aux_pt_en_translate.fst | fstconcat - compiled/aux_datareader.fst > compiled/en2pt.fst

# create datenum2text transducer
fstconcat compiled/month.fst compiled/aux_slashreader_month-day.fst | fstconcat - compiled/day.fst | fstconcat - compiled/aux_slashreader_day-year.fst | fstconcat - compiled/year.fst > compiled/datenum2text.fst

# create mix2text transducer
fstconcat compiled/aux_month2name.fst compiled/aux_slashreader_month-day.fst | fstconcat - compiled/day.fst | fstconcat - compiled/aux_slashreader_day-year.fst | fstconcat - compiled/year.fst > compiled/mix2text.fst

# create date2text transducer 
fstunion compiled/month.fst compiled/aux_month2name.fst | fstconcat - compiled/aux_slashreader_month-day.fst | fstconcat - compiled/day.fst | fstconcat - compiled/aux_slashreader_day-year.fst | fstconcat - compiled/year.fst > compiled/date2text.fst


# ######## Tests #########

# test mix2numerical with the 18y birthday dates
echo "Testing the 'mix2numerical' transducer with the input tests."
fstcompose compiled/t-96216_mix2numerical.fst compiled/mix2numerical.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96216_mix2numerical-out.fst
fstcompose compiled/t-96738_mix2numerical.fst compiled/mix2numerical.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96738_mix2numerical-out.fst

# test en2pt with the 18y birthday dates
echo "Testing the 'en2pt' transducer with the input tests."
fstcompose compiled/t-96216_en2pt.fst compiled/en2pt.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96216_en2pt-out.fst
fstcompose compiled/t-96738_en2pt.fst compiled/en2pt.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96738_en2pt-out.fst

# test datenum2text with the 18y birthday dates
echo "Testing the 'datenum2text' transducer with the input tests."
fstcompose compiled/t-96216_datenum2text_1.fst compiled/datenum2text.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96216_datenum2text_1-out.fst
fstcompose compiled/t-96216_datenum2text_2.fst compiled/datenum2text.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96216_datenum2text_2-out.fst
fstcompose compiled/t-96738_datenum2text_1.fst compiled/datenum2text.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96738_datenum2text_1-out.fst
fstcompose compiled/t-96738_datenum2text_2.fst compiled/datenum2text.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96738_datenum2text_2-out.fst

# test mix2text with the 18y birthday dates
echo "Testing the 'mix2tex' transducer with the input tests."
fstcompose compiled/t-96216_mix2text_1.fst compiled/mix2text.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96216_mix2text_1-out.fst
fstcompose compiled/t-96216_mix2text_2.fst compiled/mix2text.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96216_mix2text_2-out.fst
fstcompose compiled/t-96738_mix2text_1.fst compiled/mix2text.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96738_mix2text_1-out.fst
fstcompose compiled/t-96738_mix2text_2.fst compiled/mix2text.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96738_mix2text_2-out.fst

# test date2text with the 18y birthday dates
echo "Testing the 'date2text' transducer with the input tests."
fstcompose compiled/t-96216_date2text_1.fst compiled/date2text.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96216_date2text_1-out.fst
fstcompose compiled/t-96216_date2text_2.fst compiled/date2text.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96216_date2text_2-out.fst
fstcompose compiled/t-96216_date2text_3.fst compiled/date2text.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96216_date2text_3-out.fst
fstcompose compiled/t-96216_date2text_4.fst compiled/date2text.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96216_date2text_4-out.fst
fstcompose compiled/t-96738_date2text_1.fst compiled/date2text.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96738_date2text_1-out.fst
fstcompose compiled/t-96738_date2text_2.fst compiled/date2text.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96738_date2text_2-out.fst
fstcompose compiled/t-96738_date2text_3.fst compiled/date2text.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96738_date2text_3-out.fst
fstcompose compiled/t-96738_date2text_4.fst compiled/date2text.fst | fstshortestpath | fstproject --project_output=true | fstrmepsilon | fsttopsort > compiled/t-96738_date2text_4-out.fst


# ############ generate PDFs  ############
for fst_file in compiled/*.fst; do
    echo "Creating image: images/$(basename "$fst_file" .fst).pdf"
    fstdraw --portrait --isymbols=syms.txt --osymbols=syms.txt "$fst_file" | dot -Tpdf > "images/$(basename "$fst_file" .fst).pdf"
done


# ############      3 different ways of testing     ############
# ############ (you can use the one(s) you prefer)  ############

# 1 - generates files
# echo "\n***********************************************************"
# echo "Testing 4 (the output is a transducer: fst and pdf)"
# echo "***********************************************************"
# for i in compiled/t-*.fst; do
#     fstcompose $i compiled/mmm2mm.fst | fstshortestpath | fstproject --project_output=true |
#                   fstrmepsilon | fsttopsort > compiled/$(basename $i ".fst")-out.fst
# done
# for i in compiled/t-*-out.fst; do
# 	echo "Creating image: images/$(basename $i '.fst').pdf"
#    fstdraw --portrait --isymbols=syms.txt --osymbols=syms.txt $i | dot -Tpdf > images/$(basename $i '.fst').pdf
# done

#fstprint compiled/mix2numerical.fst

#2 - present the output as an acceptor
# trans=mmm2mm.fst
# echo "\n***********************************************************"
# echo "Testing the transducer $trans"
# echo "***********************************************************"
# for w in "SEP"; do
#     echo "\t $w"
#     python3 ./scripts/word2fst.py $w | fstcompile --isymbols=syms.txt --osymbols=syms.txt | fstarcsort |
#                      fstcompose - compiled/$trans | fstshortestpath | fstproject --project_output=true |
#                      fstrmepsilon | fsttopsort | fstprint --acceptor --isymbols=syms.txt
#done
#to generate the txt files for each transducer!!!!!!
#done

#3 - presents the output with the tokens concatenated (uses a different syms on the output)
# fst2word() {
# 	awk '{if(NF>=3){printf("%s",$3)}}END{printf("\n")}'
# }

# trans=date2text.fst
# echo "\n***********************************************************"
# echo "Testing $trans"
# echo "***********************************************************"
# for w in "APR/18/2019" "4/18/2019" "04/18/2019" "ABR/18/2019"; do
#     res=$(python3 ./scripts/word2fst.py $w | fstcompile --isymbols=syms.txt --osymbols=syms.txt | fstarcsort |
#                        fstcompose - compiled/$trans | fstshortestpath | fstproject --project_output=true |
#                        fstrmepsilon | fsttopsort | fstprint --acceptor --isymbols=./scripts/syms-out.txt | fst2word)
#     echo "$w = $res"
# done

echo "The end"