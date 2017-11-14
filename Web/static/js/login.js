$(document).ready(function () {
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
});
