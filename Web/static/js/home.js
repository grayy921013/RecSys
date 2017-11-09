$(document).ready(function() {
    $(".popoverData").popover();

    // check if visited
    $.get("/check_visited?path=index", function(data){
        if(data == "true"){
            alert("true")
        } else if(data == "false"){
            alert("false")
            // popup
            $('#myModal').modal()

            // set that page as visited
            $.get("/visit_page?path=index")
        }
    })
});