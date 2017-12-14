-- rollback;
-- begin;

drop table pivoted_2;

select colpivot('pivoted_2', 'select b.movielens_id movie_id, c.movielens_id similar_movie_id, a.rank, a.algorithm
                from mainsite_similarmovie a
                left join mainsite_movie b on a.movie_id = b.id
                left join mainsite_movie c on a.similar_movie_id = c.id
                order by algorithm, movie_id',
    array['movie_id', 'algorithm'], array['rank'], '#.similar_movie_id', null);

-- select * from pivoted_2 order by algorithm, movie_id;

Copy (select * from pivoted_2) To 'C:/tmp/output.csv' With CSV DELIMITER ',';
-- rollback;