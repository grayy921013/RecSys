var setupCSRF = function () {
    // using jQuery
    function getCookie(name) {
        var cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            var cookies = document.cookie.split(';');
            for (var i = 0; i < cookies.length; i++) {
                var cookie = jQuery.trim(cookies[i]);
                // Does this cookie string begin with the name we want?
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    var csrftoken = getCookie('csrftoken');

    function csrfSafeMethod(method) {
        // these HTTP methods do not require CSRF protection
        return (/^(GET|HEAD|OPTIONS|TRACE)$/.test(method));
    }

    $.ajaxSetup({
        beforeSend: function (xhr, settings) {
            if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        }
    });
}


$(document).ready(function () {
    setupCSRF();

    $(".popoverData").popover();

    var ui_movie_list = $(".movie-item");
    var target_movie_id = $(".target-movie-holder").attr('id');

    var movie_list = [];

    // Debug code.
    target_movie_id = '3';

    var bindClickListener = function (movie_id, action) {
        console.log("click " + action + " movie Id " + movie_id);
        $.post("/uservote",
            {
                movie1_id: target_movie_id,
                movie2_id: movie_id,
                action: action
            },
            function (data, status) {
                console.log("vote status " + status);
                console.log(data);

                var rest_list = []
                for (var i = 0; i < movie_list.length; i++) {
                    if (movie_list[i].id == movie_id) {
                        console.log("remove index " + i);
                    } else {
                        rest_list.push(movie_list[i])
                    }
                }
                movie_list = rest_list;
                bindMovie(movie_list)
            });
    }

    var bindMovie = function (movie_list) {
        for (var i = 0; i < ui_movie_list.length; i++) {
            var uiHolder = ui_movie_list[i];
            var uiTitle = $($(uiHolder).find(".similar-title")).find("b");
            var uiPopoverData = $(uiHolder).find(".popoverData");
            var uiImage = $(uiPopoverData).find("img");
            var uiButtonHolder = $(uiHolder).find(".label-bar");
            var uiButtonNo = $(uiButtonHolder).find(".btn-no");
            var uiButtonSkip = $(uiButtonHolder).find(".btn-skip");
            var uiButtonYes = $(uiButtonHolder).find(".btn-yes");

            if (i < movie_list.length) {
                // set the holder to be visible and bind data
                $(uiHolder).show();

                var movie = movie_list[i];

                uiTitle.text(movie.title + " (" + movie.year + ")");
                if (movie.plot) {
                    uiPopoverData.attr("data-content", movie.plot);
                } else {
                    uiPopoverData.attr("data-content", "No plot available");
                }
                uiImage.attr("src", movie.poster);

                $(uiButtonNo).attr("id", "btn-no---" + movie.id);
                $(uiButtonSkip).attr("id", "btn-skip-" + movie.id);
                $(uiButtonYes).attr("id", "btn-yes--" + movie.id);

                $(uiButtonNo).unbind();
                $(uiButtonNo).click(function () {
                    var id = $(this).attr("id").substring(9);
                    bindClickListener(id, -1);
                })

                $(uiButtonSkip).unbind();
                $(uiButtonSkip).click(function () {
                    var id = $(this).attr("id").substring(9);
                    bindClickListener(id, 0);
                })

                $(uiButtonYes).unbind();
                $(uiButtonYes).click(function () {
                    var id = $(this).attr("id").substring(9);
                    bindClickListener(id, 1);
                })

            } else {
                // set the holder to be invisible
                $(uiHolder).hide()
            }
        }
    };


    var getSimilarList = function () {
        $.get("/getsimilar/" + target_movie_id, function (data) {
            console.log("get similar movie ");
            console.log(data);
            movie_list = data.data;
            bindMovie(data.data);
        });
    };

    console.log("get target movie id " + target_movie_id);
    getSimilarList();

});
