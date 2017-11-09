$(document).ready(function() {
    $(".popoverData").popover();

    // check if visited
    $.get("/check_visited?path=label", function(data){
        if(data == "true"){
            alert("true")
        } else if(data == "false"){
            alert("false")
            // popup
            $('#myModal').modal()

            // set that page as visited
            $.get("/visit_page?path=label")
        }
    })
});
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

    var ui_movie_list = $(".movie-item");
    var target_movie_id = $(".target-movie-holder").attr('id');
    var label_progress_text = $("#label-progress")
    var label_progress_bar = $("#label-progress-bar")

    var movie_list = [];

    var refreshLabelProgress = function (movie_list) {
        var total = 0;
        var labeled = 0;
        for (var i = 0; i < movie_list.length; i++) {
            var movie = movie_list[i];
            total += 1;
            if (movie.status != 2) {
                labeled += 1;
            }
        }

        label_progress_text.text("Progress: " + labeled + "/" + total);
        var percent = 100;
        if (total > 0) {
            percent = labeled / total * 100;
        }
        label_progress_bar.attr('style', "width: " + percent + "%");
    }


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

                for (var i = 0; i < movie_list.length; i++) {
                    if (movie_list[i].id == movie_id) {
                        movie_list[i].status = action;
                        refreshMovie(movie_id)
                        refreshLabelProgress(movie_list);
                    }
                }
            });
    }

    var refreshMovie = function (movie_id) {
        for (var i = 0; i < ui_movie_list.length; i++) {
            var uiHolder = ui_movie_list[i];
            var uiButtonHolder = $(uiHolder).find(".label-bar");
            var uiButtonNo = $(uiButtonHolder).find(".btn-no");
            var uiButtonSkip = $(uiButtonHolder).find(".btn-skip");
            var uiButtonYes = $(uiButtonHolder).find(".btn-yes");

            if (i < movie_list.length && movie_list[i].id == movie_id) {
                var movie = movie_list[i];

                // bind rating status
                if (movie.status != 2) {
                    $(uiHolder).addClass('animation-target')
                    $(uiHolder).fadeTo(300, 0.5);

                    switch (movie.status) {
                        case -1:
                            uiButtonNo.addClass('btn-glow');
                            uiButtonSkip.removeClass('btn-glow');
                            uiButtonYes.removeClass('btn-glow');
                            break;
                        case 0:
                            uiButtonNo.removeClass('btn-glow');
                            uiButtonSkip.addClass('btn-glow');
                            uiButtonYes.removeClass('btn-glow');
                            break;
                        case 1:
                            uiButtonNo.removeClass('btn-glow');
                            uiButtonSkip.removeClass('btn-glow');
                            uiButtonYes.addClass('btn-glow');
                            break;
                    }

                } else {
                    $(uiHolder).fadeTo(0, 1);
                    $(uiHolder).removeClass('animation-target')
                }
            }
        }
    }

    var bindMovie = function (movie_list) {
        for (var i = 0; i < ui_movie_list.length; i++) {
            var uiHolder = ui_movie_list[i];
            var uiTitleAnchor = $($(uiHolder).find(".popoverData"));
            var uiTitle = $($(uiHolder).find(".movie-title"));
            var uiYear= $($(uiHolder).find(".movie-year"));
            var uiPopoverData = $(uiHolder).find(".movie-hover-text");
            var uiImage = $(uiHolder).find("img");
            var uiButtonHolder = $(uiHolder).find(".label-bar");
            var uiButtonNo = $(uiButtonHolder).find(".btn-no");
            var uiButtonSkip = $(uiButtonHolder).find(".btn-skip");
            var uiButtonYes = $(uiButtonHolder).find(".btn-yes");
            var uiButtonShow = $(uiHolder).find(".btn-show");
            var uiOverlay = $(uiHolder).find(".movie-overlay");

            if (i < movie_list.length) {
                // set the holder to be visible and bind data
                $(uiHolder).show();

                var movie = movie_list[i];

                // bind content

                uiTitleAnchor.attr("data-content", movie.title);
                uiTitle.text(movie.title);
                uiYear.text("(" + movie.year + ")");
                if (movie.plot) {
                    uiPopoverData.text(movie.plot)
                } else {
                    uiPopoverData.text("No plot available");
                }
                uiImage.attr("src", movie.poster);

                $(uiButtonNo).attr("id", "btn-no---" + movie.id);
                $(uiButtonSkip).attr("id", "btn-skip-" + movie.id);
                $(uiButtonYes).attr("id", "btn-yes--" + movie.id);

                $(uiButtonNo).unbind();
                $(uiButtonNo).click(function () {
                    if ($(this).attr("class").indexOf("btn-glow") == -1) {
                        $(this).blur();
                        var id = $(this).attr("id").substring(9);
                        bindClickListener(id, -1);
                    } else {
                        $(this).removeClass("btn-glow");
                        var id = $(this).attr("id").substring(9);
                        bindClickListener(id, 2);
                    }
                })

                $(uiButtonSkip).unbind();
                $(uiButtonSkip).click(function () {
                    if ($(this).attr("class").indexOf("btn-glow") == -1) {
                        $(this).blur();
                        var id = $(this).attr("id").substring(9);
                        bindClickListener(id, 0);
                    } else {
                        $(this).removeClass("btn-glow");
                        var id = $(this).attr("id").substring(9);
                        bindClickListener(id, 2);
                    }
                })

                $(uiButtonYes).unbind();
                $(uiButtonYes).click(function () {
                    if ($(this).attr("class").indexOf("btn-glow") == -1) {
                        $(this).blur();
                        var id = $(this).attr("id").substring(9);
                        bindClickListener(id, 1);
                    } else {
                        $(this).removeClass("btn-glow");
                        var id = $(this).attr("id").substring(9);
                        bindClickListener(id, 2);
                    }
                })

                // bind rating status
                if (movie.status != 2) {
                    $(uiHolder).fadeTo(0, 0.5);

                    switch (movie.status) {
                        case -1:
                            uiButtonNo.addClass('btn-glow');
                            uiButtonSkip.removeClass('btn-glow');
                            uiButtonYes.removeClass('btn-glow');
                            break;
                        case 0:
                            uiButtonNo.removeClass('btn-glow');
                            uiButtonSkip.addClass('btn-glow');
                            uiButtonYes.removeClass('btn-glow');
                            break;
                        case 1:
                            uiButtonNo.removeClass('btn-glow');
                            uiButtonSkip.removeClass('btn-glow');
                            uiButtonYes.addClass('btn-glow');
                            break;
                    }

                } else {
                    $(uiHolder).fadeTo(0, 1);
                }

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
            refreshLabelProgress(movie_list);
        });
    };

    console.log("get target movie id " + target_movie_id);
    getSimilarList();

});
