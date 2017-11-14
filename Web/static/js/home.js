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
        $('#' + hide_img_id).hide();
        $('#' + show_img_id).show();
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
    $.get("/check_visited?path=index", function (data) {
        if (data == "true") {
        } else if (data == "false") {
            // popup
            initGallery();
            modal_selector.modal()

            // set that page as visited
            $.get("/visit_page?path=index")
        }
    })

    var previousLabelCount = 0;
    // update goal
    setInterval(function () {
        $.get("/get_vote_count", function (data) {
            var currentLabelCount = parseInt(data)
            $('.nav-goal').removeClass("animated bounceIn");
            if (currentLabelCount != previousLabelCount) {
                $(".nav-goal").text("Our Goal: " + currentLabelCount + "/2000")
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

    function updateGallery(selector) {
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
