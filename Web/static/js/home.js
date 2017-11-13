$(document).ready(function () {
    $(".popoverData").popover();

    var current_img = 1;
    var total_img = 3;

    initGallery();

    $('#show_prev').click(function () {
        var hide_img_id = "img_" + current_img;
        current_img--;
        var show_img_id = "img_" + current_img;
        $('#' + hide_img_id).hide();
        $('#' + show_img_id).show();
        updateGallery();
    });

    $('#show_next').click(function () {
        if (current_img >= total_img) {
            $('#helpModal').modal('hide');
            initGallery();
        }
        else {
            var hide_img_id = "img_" + current_img;
            current_img++;
            var show_img_id = "img_" + current_img;
            $('#' + hide_img_id).hide();
            $('#' + show_img_id).show();
            updateGallery();
        }
    });

    // check if visited
    $.get("/check_visited?path=index", function (data) {
        if (data == "true") {
        } else if (data == "false") {
            // popup
            initGallery();
            $('#helpModal').modal()

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
                $(".nav-goal").text("Our Goal: " + currentLabelCount + "/100")
                $('.nav-goal').addClass("animated bounceIn");

                previousLabelCount = currentLabelCount;
            }
        })
    }, 1000);

    function initGallery() {
        current_img = 1;
        $("#show_prev").hide();
        $("#show_next").text("Next");
        $('#img_1').show();
        $('#img_2').hide();
        $('#img_3').hide();
    }

    function updateGallery(selector) {
        if (current_img >= total_img) {
            $("#show_next").text("OK");
        }
        else if (current_img <= 1) {
            $("#show_prev").hide();
        }
        else {
            $("#show_prev").show();
        }
    }

});
