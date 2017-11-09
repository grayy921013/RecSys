$(document).ready(function() {
    $(".popoverData").popover();

    // check if visited
    $.get("/check_visited?path=index", function(data){
        if(data == "true"){
        } else if(data == "false"){
            // popup
            $('#myModal').modal()

            // set that page as visited
            $.get("/visit_page?path=index")
        }
    })
});
