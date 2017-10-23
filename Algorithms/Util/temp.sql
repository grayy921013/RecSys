create table mainsite_similarityals_bk as
select m1.id as id1_id, m2.id as id2_id, s.als_cosine
from mainsite_similarityals s
join  mainsite_movie m1 on s.id1_id = m1.movielens_id
join  mainsite_movie m2 on s.id2_id = m2.movielens_id;

truncate table mainsite_similarityals;

insert into mainsite_similarityals
select *
from mainsite_similarityals_bk;

drop table mainsite_similarityals_bk;
