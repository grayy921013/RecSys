$(document).ready(function () {
    $(".popoverData").popover();

    // check if visited
    $.get("/check_visited?path=index", function (data) {
        if (data == "true") {
        } else if (data == "false") {
            // popup
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

});
