alter table mainsite_similarityals
 add column db_id1 int, 
 add column db_id2 int;

update mainsite_similarityals
   set db_id1 = m.id
  from mainsite_movie m
 where id1_id = m.movielens_id;

update mainsite_similarityals
   set db_id2 = m.id
  from mainsite_movie m
 where id2_id = m.movielens_id;


delete from mainsite_similarityals
 where db_id2 is null 
    or db_id1 is null;


update mainsite_similarity
   set year = (a.year-b.year)*(a.year-b.year)
  from movie_details a, movie_details b
 where id1_id = a.id
   and id2_id = b.id;

-- UPDATE 22179941
-- Query returned successfully in 33 min.

-- create table mainsite_similarityals_bk as
-- select m1.id as id1_id, m2.id as id2_id, s.als_cosine
-- from mainsite_similarityals s
-- join  mainsite_movie m1 on s.id1_id = m1.movielens_id
-- join  mainsite_movie m2 on s.id2_id = m2.movielens_id;

-- truncate table mainsite_similarityals;

-- insert into mainsite_similarityals
-- select *
-- from mainsite_similarityals_bk;

-- drop table mainsite_similarityals_bk;
