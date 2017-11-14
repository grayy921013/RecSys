$(document).ready(function () {
    $(".popoverData").popover();

    var current_img = 1;
    var total_img = 4;
    var screen_width = screen.width;
    var modal_selector;

    if (screen_width >= 768) {
        console.log("set to help modal")
        modal_selector = $('#helpModal');
        $('#nav_help_button').attr('data-target','#helpModal');
    }
    else {
        console.log("set to MOBILE help modal")
        modal_selector = $('#mobileHelpModal');
        $('#nav_help_button').attr('data-target','#mobileHelpModal');
    }

    initGallery();

    $('#show_prev', modal_selector).click(function () {
        var hide_img_id = "img_" + current_img;
        current_img--;
        var show_img_id = "img_" + current_img;
        $('#' + hide_img_id, modal_selector).hide();
        $('#' + show_img_id, modal_selector).show();
        updateGallery();
    });

    $('#show_next', modal_selector).click(function () {
        if (current_img >= total_img) {
            modal_selector.modal('hide');
            initGallery();
        }
        else {
            var hide_img_id = "img_" + current_img;
            current_img++;
            var show_img_id = "img_" + current_img;
            $('#' + hide_img_id, modal_selector).hide();
            $('#' + show_img_id, modal_selector).show();
            updateGallery();
        }
    });

    // check if visited
    $.get("/check_visited?path=label", function (data) {
        if (data == "true") {
        } else if (data == "false") {
            // popup
            initGallery();
            modal_selector.modal()

            // set that page as visited
            $.get("/visit_page?path=label")
        }
    })

    var previousLabelCount = 0;
    // update goal
    setInterval(function () {
        $.get("/get_vote_count", function (data) {
            var currentLabelCount = parseInt(data)
            $('.nav-goal').removeClass("animated bounceIn");
            if (currentLabelCount != previousLabelCount) {
                $(".nav-goal").text("Our Goal: " + currentLabelCount + "/5000")
                $('.nav-goal').addClass("animated bounceIn");

                previousLabelCount = currentLabelCount;
            }
        })
    }, 1000);

    function initGallery() {
        current_img = 1;
        $("#show_prev", modal_selector).hide();
        $("#show_next", modal_selector).text("Next");
        $('#img_1', modal_selector).show();
        $('#img_2', modal_selector).hide();
        $('#img_3', modal_selector).hide();
        $('#img_4', modal_selector).hide();
    }

    function updateGallery() {
        if (current_img >= total_img) {
            $("#show_next", modal_selector).text("OK");
        }
        else if (current_img <= 1) {
            $("#show_prev", modal_selector).hide();
        }
        else {
            $("#show_prev", modal_selector).show();
        }
    }

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
    var label_progress_bar = $("#label-progress")

    var movie_list = [];

    var lastPercent = -1;

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

        label_progress_bar.text(labeled + "/" + total);
        var percent = 100;
        if (total > 0) {
            percent = labeled / total * 100;
        }
        label_progress_bar.attr('style', "width: " + percent + "%");

        if (percent == 100 && lastPercent < 100 && lastPercent >= 0) {
            startAnimation()
        }

        lastPercent = percent;
    };


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
            var uiYear = $($(uiHolder).find(".movie-year"));
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

    $(".navbar-back").on('click', function () {
        window.history.back();
    });


    // setup animation

    $(function () {
        var numberOfStars = 20;

        for (var i = 0; i < numberOfStars; i++) {
            $('.congrats').append('<div class="blob fa fa-star ' + i + '"></div>');
        }

        animateText();

        animateBlobs();
    });

    var startAnimation = function () {
        $(".mask-holder").show();
        reset();

        animateText();

        animateBlobs();
    }

    function reset() {
        $.each($('.blob'), function (i) {
            TweenMax.set($(this), {x: 0, y: 0, opacity: 1});
        });

        TweenMax.set($('h1'), {scale: 1, opacity: 1, rotation: 0});
    }

    function animateText() {
        TweenMax.from($('h1'), 0.8, {
            scale: 0.4,
            opacity: 0,
            rotation: 15,
            ease: Back.easeOut.config(4),
        });
    }

    function animateBlobs() {

        var xSeed = _.random(350, 380);
        var ySeed = _.random(120, 170);

        $.each($('.blob'), function (i) {
            var $blob = $(this);
            var speed = _.random(1, 5);
            var rotation = _.random(5, 100);
            var scale = _.random(0.8, 1.5);
            var x = _.random(-xSeed, xSeed);
            var y = _.random(-ySeed, ySeed);

            TweenMax.to($blob, speed, {
                x: x,
                y: y,
                ease: Power1.easeOut,
                opacity: 0,
                rotation: rotation,
                scale: scale,
                onStartParams: [$blob],
                onStart: function ($element) {
                    $element.css('display', 'block');
                },
                onCompleteParams: [$blob],
                onComplete: function ($element) {
                    $element.css('display', 'none');
                }
            });
        });
    }

});
