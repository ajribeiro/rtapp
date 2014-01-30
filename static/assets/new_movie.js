
function setup_writer_fill(row){
    $('.example-writers'+row.toString()+' .typeahead').typeahead({name: 'writerlist',prefetch: 'static/writerlist.json',limit: 10})
}

var wrowcnt = 0;
$(document).ready(function(){
    var writerhtml = "<td class='writername'><div class='example-writers"+wrowcnt.toString()+"'>"
    writerhtml += "<span class='twitter-typeahead' style='position: relative; display: inline-block; '>"
    writerhtml += "<input class='typeahead tt-query writername' type='text' placeholder='writer' "
    writerhtml += "autocomplete='off' spellcheck='false' style='position: relative; vertical-align: top; background-color: transparent; ' dir='auto' name='title' id='title'>"
    writerhtml += "</span></div></td>"
    $('#writerbody').append('<tr>')
    $('#writerbody').append(writerhtml)
    $('#writerbody').append(writerhtml)
    $('#writerbody').append(writerhtml)
    $('#writerbody').append('</tr>')
    setup_writer_fill(wrowcnt);
})

$(document).ready(function(){
    $('#writeradd').click(function(){
        wrowcnt ++;
    var writerhtml = "<td class='writername'><div class='example-writers"+wrowcnt.toString()+"'>"
    writerhtml += "<span class='twitter-typeahead' style='position: relative; display: inline-block; '>"
    writerhtml += "<input class='typeahead tt-query writername' type='text' placeholder='writer' "
    writerhtml += "autocomplete='off' spellcheck='false' style='position: relative; vertical-align: top; background-color: transparent; ' dir='auto' name='title' id='title'>"
    writerhtml += "</span></div></td>"
        $('#writerbody').append('<tr>')
        $('#writerbody').append(writerhtml)
        $('#writerbody').append(writerhtml)
        $('#writerbody').append(writerhtml)
        $('#writerbody').append('</tr>')
        setup_writer_fill(wrowcnt);
    })
})




function setup_actor_fill(row){
    $('.example-actors'+row.toString()+' .typeahead').typeahead({name: 'actorlist',prefetch: 'static/actorlist.json',limit: 10})
}

var arowcnt = 0;
$(document).ready(function(){
    var actorhtml = "<td><div class='example-actors"+arowcnt.toString()+"'>"
    actorhtml += "<span class='twitter-typeahead' style='position: relative; display: inline-block; '>"
    actorhtml += "<input class='typeahead tt-query actorname' type='text' placeholder='Actor' "
    actorhtml += "autocomplete='off' spellcheck='false' style='position: relative; vertical-align: top; background-color: transparent; ' dir='auto' name='title' id='title'>"
    actorhtml += "</span></div></td>"
    $('#actorbody').append('<tr>')
    $('#actorbody').append(actorhtml)
    $('#actorbody').append(actorhtml)
    $('#actorbody').append(actorhtml)
    $('#actorbody').append('</tr>')
    setup_actor_fill(arowcnt);
})

$(document).ready(function(){
    $('#actoradd').click(function(){
        arowcnt ++;
        var actorhtml = "<td class='actorname'><div class='example-actors"+arowcnt.toString()+"'>"
        actorhtml += "<span class='twitter-typeahead' style='position: relative; display: inline-block; '>"
        actorhtml += "<input class='typeahead tt-query actorname' type='text' placeholder='Actor' "
        actorhtml += "autocomplete='off' spellcheck='false' style='position: relative; vertical-align: top; background-color: transparent; ' dir='auto' name='title' id='title'>"
        actorhtml += "</span></div></td>"
            $('#actorbody').append('<tr>')
            $('#actorbody').append(actorhtml)
            $('#actorbody').append(actorhtml)
            $('#actorbody').append(actorhtml)
            $('#actorbody').append('</tr>')
            setup_actor_fill(arowcnt);
    })
})

$(document).ready(function(){
    $('#mybutton').click(function(){
        var actors = [], writers = [], directors = []

        var elemlist = $('.actorname')
        for(var k=0; k<elemlist.length; k++){
            var name = elemlist[k].value
            if(name != '' && name != undefined)
                actors.push(name)
        }
        var elemlist = $('.writername')
        for(var k=0; k<elemlist.length; k++){
            var name = elemlist[k].value
            if(name != '' && name != undefined)
                writers.push(name)
        }
        var elemlist = $('.directorname')
        for(var k=0; k<elemlist.length; k++){
            var name = elemlist[k].value
            if(name != '' && name != undefined)
                directors.push(name)
        }


    $.ajax({
        type: 'POST',
        url: 'recalc',
        contentType: 'application/json',
        data: JSON.stringify({'actors': actors,'director':directors,'writers':writers}),
        dataType: 'json',
        success: function(result){
            console.log(result)
            var lastval = -1
            if($('#predrat')[0].innerText != 'N/A') 
                lastval = $('#predrat')[0].innerText
            $('#predrat')[0].innerText = result.result.toString()

            if(result.result > lastval)
                $('#predrat').css('color','green')
            else if(result.result < lastval)
                $('#predrat').css('color','red')

            setTimeout(function(){
                $('#predrat').css('color','#333333')
            },1500);

            if(result.result >= 60){
                $('#typeicon')[0].src = '/static/fresh.png'
            }
            else{
                $('#typeicon')[0].src = '/static/rotten.png'
            }
            $('#typeicon')[0].width=50
        }
    });


    })
})
