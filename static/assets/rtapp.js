var fileslist = []
var titlelist = []
var actornames = []
var actorids = []
var writernames = []
var writerids = []
var directors = []
var i=0
var viewheight
var currentMousePos = { x: -1, y: -1 };

function recalculate(dd){
    $.ajax({
        type: 'POST',
        url: 'recalc',
        contentType: 'application/json',
        data: JSON.stringify(dd),
        dataType: 'json',
        success: function(result){
            setTimeout(function(){
                loadhtml = "<div id='load' class='loadingdiv'><img src='static/ajax-loader.gif'><p>Recalculating...</p></div>"

                // $('body').append(loadhtml)


                value = $('#predrat')[0].innerHTML
                if(result['result'] > value) color = 'green'
                else if(result['result'] < value) color = 'red'

                $('#predrat')[0].innerHTML = result['result'].toString()
                $('#predrat').css('color',color)
                
                setTimeout(function(){
                    $('#predrat').css('color','#333333')
                },1500);

                console.log(result['result'])
                if(result['result'] > 59){
                    $('#predicon')[0].innerHTML = "<img src='static/fresh.png'>"
                    $('#predtext')[0].innerHTML = 'Fresh!'
                }
                else{
                    $('#predicon')[0].innerHTML = "<img src='static/rotten.png'>"
                    $('#predtext')[0].innerHTML = 'Rotten!'
                }
                // $('#load').remove()
            },0)
        }
    });
}



function load_more_posters(){
    for(var j=i; j<i+3; j++){
        $('#posterlist').append("<a href='/search_movie?title="+titlelist[j]+"'>"+
            "<img src="+fileslist[j]+" class='movieposter' width='243px' height='350px'>"+
            "</a>"
        )
    }
    i += 3
}

jQuery(function($) {
    $(document).mousemove(function(event) {
        currentMousePos.x = event.pageX;
        currentMousePos.y = event.pageY;
    });
});



$.ajax({
    dataType: "json",
    url: 'static/posterlist.json',
    success: function(result){
        viewheight = $(window).height()

        for(var j=0; j<result.length; j++){
            titlelist.push(result[j][0])
            fileslist.push(result[j][1])
        }
        // fileslist = result;
        $(window).scroll(function() {
           if($(window).scrollTop() + viewheight >= $(document).height()) {
              load_more_posters()
            }
        })
        posterlist = []
        for(var j=0; j<3; j++){
            link = "\"/search_movie?title=";
            link = link.concat(titlelist[j])
            var titleid = titlelist[j].replace(/ /g,'').replace(/:/g,'').replace(/'/g,'')
            $('#posterlist').append("<a href="+link+"\">"+
                "<img id="+titleid+" src="+fileslist[j]+" class='movieposter' width='243px' height='350px' class='posimg'>"+
                "</a>"
            )
            posterlist.push('#'+titleid)
        }
        i += 3
    }
});

function load_actorlist(){
    $.ajax({
        dataType: 'json',
        url: '/static/actorlist.json',
        success: function(result){
            for(var k=0; k<result.length;k++)
            {
                actorids.push(result[k][0])
                actornames.push(result[k][1]);
            }
        }
    })
}
function load_writerlist(){
    $.ajax({
        dataType: 'json',
        url: '/static/writerlist.json',
        success: function(result){
            for(var k=0; k<result.length;k++)
            {
                writerids.push(result[k][0])
                writernames.push(result[k][1]);
            }
        }
    })
}

function load_directorlist(){
    $.ajax({
        dataType: 'json',
        url: '/static/directorlist.json',
        success: function(result){
            for(var k=0; k<result.length;k++)
            {
                directors.push(result[k]);
            }
        }
    })
}
