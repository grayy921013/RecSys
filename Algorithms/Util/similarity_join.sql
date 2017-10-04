CREATE OR REPLACE FUNCTION SIMILARITY_JOIN()
RETURNS void AS $$
BEGIN

    -- Join all the tables in a single temporary one
    -- Disclaimer: We are using PLOT as the anchor, since the plot is the field with the biggest amount of pairs > 0
    --             However the official solution should be more comprehensive, looking for a efficient way to show all the pairs from all the tables
    CREATE TABLE TMP AS
    SELECT  a.id1_id, 
            a.id2_id, 
            title_tfitf,
            title_bm25,
            title_jaccard,
            genre_tfitf,
            genre_bm25,
            genre_jaccard,
            released_tfitf,
            released_bm25,
            released_jaccard,
            director_tfitf,
            director_bm25,
            director_jaccard,
            writer_tfitf,
            writer_bm25,
            writer_jaccard,
            cast_tfitf,
            cast_bm25,
            cast_jaccard,
            metacritic_tfitf,
            metacritic_bm25,
            metacritic_jaccard,
            plot_tfitf,
            plot_bm25,
            plot_jaccard,
            full_plot_tfitf,
            full_plot_bm25,
            full_plot_jaccard,
            language_tfitf,
            language_bm25,
            language_jaccard,
            country_tfitf,
            country_bm25,
            country_jaccard,
            awards_tfitf,
            awards_bm25,
            awards_jaccard,
            last_updated_tfitf,
            last_updated_bm25,
            last_updated_jaccard
    FROM mainsite_SimilarityFull_plot as a
    LEFT OUTER JOIN mainsite_SimilarityGenre        as c on a.id1_id = c.id1_id and a.id2_id = c.id2_id
    LEFT OUTER JOIN mainsite_SimilarityReleased     as d on a.id1_id = d.id1_id and a.id2_id = d.id2_id
    LEFT OUTER JOIN mainsite_SimilarityDirector     as b on a.id1_id = b.id1_id and a.id2_id = b.id2_id
    LEFT OUTER JOIN mainsite_SimilarityWriter       as e on a.id1_id = e.id1_id and a.id2_id = e.id2_id
    LEFT OUTER JOIN mainsite_SimilarityCast         as f on a.id1_id = f.id1_id and a.id2_id = f.id2_id
    LEFT OUTER JOIN mainsite_SimilarityMetacritic   as g on a.id1_id = g.id1_id and a.id2_id = g.id2_id
    LEFT OUTER JOIN mainsite_SimilarityPlot         as h on a.id1_id = h.id1_id and a.id2_id = h.id2_id
    LEFT OUTER JOIN mainsite_SimilarityTitle        as i on a.id1_id = i.id1_id and a.id2_id = i.id2_id
    LEFT OUTER JOIN mainsite_SimilarityLanguage     as j on a.id1_id = j.id1_id and a.id2_id = j.id2_id
    LEFT OUTER JOIN mainsite_SimilarityCountry      as k on a.id1_id = k.id1_id and a.id2_id = k.id2_id
    LEFT OUTER JOIN mainsite_SimilarityAwards       as l on a.id1_id = l.id1_id and a.id2_id = l.id2_id
    LEFT OUTER JOIN mainsite_SimilarityLast_updated as m on a.id1_id = m.id1_id and a.id2_id = m.id2_id;

    -- Insert data into legit table
    INSERT INTO mainsite_similarity(
           id1_id, id2_id, title_tfitf, title_bm25, title_jaccard, genre_tfitf, genre_bm25, genre_jaccard, released_tfitf, released_bm25, released_jaccard, director_tfitf, director_bm25, director_jaccard, writer_tfitf, writer_bm25, writer_jaccard, cast_tfitf, cast_bm25, cast_jaccard, metacritic_tfitf, metacritic_bm25, metacritic_jaccard, plot_tfitf, plot_bm25, plot_jaccard, full_plot_tfitf, full_plot_bm25, full_plot_jaccard, language_tfitf, language_bm25, language_jaccard, country_tfitf, country_bm25, country_jaccard, awards_tfitf, awards_bm25, awards_jaccard, last_updated_tfitf, last_updated_bm25, last_updated_jaccard)  
    SELECT id1_id, id2_id, title_tfitf, title_bm25, title_jaccard, genre_tfitf, genre_bm25, genre_jaccard, released_tfitf, released_bm25, released_jaccard, director_tfitf, director_bm25, director_jaccard, writer_tfitf, writer_bm25, writer_jaccard, cast_tfitf, cast_bm25, cast_jaccard, metacritic_tfitf, metacritic_bm25, metacritic_jaccard, plot_tfitf, plot_bm25, plot_jaccard, full_plot_tfitf, full_plot_bm25, full_plot_jaccard, language_tfitf, language_bm25, language_jaccard, country_tfitf, country_bm25, country_jaccard, awards_tfitf, awards_bm25, awards_jaccard, last_updated_tfitf, last_updated_bm25, last_updated_jaccard
        FROM public.tmp;

    -- Delete fake table
    DROP TABLE public.tmp;

    -- Confirm you have the data
    -- TODO: Return this value
    -- SELECT count(*)
    --   FROM mainsite_similarity;
END;
$$ LANGUAGE plpgsql;
