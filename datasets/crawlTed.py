from apiclient.discovery import build
import datetime
import re
from youtube_transcript_api import YouTubeTranscriptApi
import os
from nltk import word_tokenize
from nltk.stem import SnowballStemmer

def get_channel_videos(channel_id):

    res = youtube.channels().list(id=channel_id,
                                  part='contentDetails').execute()
    playlist_id = res['items'][0]['contentDetails']['relatedPlaylists']['uploads']

    videos = []
    next_page_token = None

    while 1:
        res = youtube.playlistItems().list(playlistId=playlist_id,
                                           part='snippet',
                                           maxResults=50,
                                           pageToken=next_page_token).execute()

        videos += res['items']
        next_page_token = res.get('nextPageToken')

        if next_page_token is None:
            break
    return videos


def scrapyt(apikey,oppath,listofvid_path):
    youtube = build('youtube', 'v3', developerKey=apikey)
    videos = get_channel_videos('UCAuUUnT6oDeKwE6v1NGQxug')

    lvid = len(videos)
    url=[]
    missed = {}
    cnt=0
    lang_list = ['en','fr','es']
    stem_es = SnowballStemmer('spanish')
    stem_fr = SnowballStemmer('french')
    stem_en = SnowballStemmer('english')
    for i in videos[:100]:
        cnt+=1
        print("looping ...",str(cnt)," out of  ",str(lvid))

        url.append(i['snippet']['resourceId']['videoId'])
    #     title.append(i['snippet']['title'])
        video_id = i['snippet']['resourceId']['videoId']

        for lang in lang_list:
            try:
                # print(video_id)
                trans = YouTubeTranscriptApi.get_transcript(video_id ,languages=[lang])

                counter = 1
                oppath1 = oppath  +video_id+'_'+lang+'.srt'
                with open(oppath1, 'w') as the_file:
                    for t in trans:
                        end = int(t['start']) + int(t['duration'])
                        the_file.write( str(counter) + '\n' )
                        the_file.write( str(datetime.timedelta(seconds=t['start'])).rstrip("0") + ',000 --> '+
                                       str(datetime.timedelta(seconds= end) ).rstrip("0") +',000\n' )
                        # the_file.write( ' '.join( t['text'].split())  + '\n' )
                        if lang == 'en':
                            the_file.write( " ".join( [stem_en.stem(i) for i in re.sub(r"[^A-Za-z ]+", '', t['text']).split()] )  + '\n' )
                        if lang == 'es':
                            the_file.write( " ".join( [stem_es.stem(i) for i in re.sub(r"[^A-Za-z ]+", '', t['text']).split()] )  + '\n' )
                        if lang == 'fr':
                            the_file.write( " ".join( [stem_fr.stem(i) for i in re.sub(r"[^A-Za-z ]+", '', t['text']).split()] )  + '\n' )
                        the_file.write( '\n' )

                        counter +=1
            # except:
            except Exception as e:
                # print(e)
                print('except  ',video_id, '   ',lang )
                if lang not in missed:
                    temp= []
                    temp.append(video_id)
                    missed[lang] = temp
                else:
                    temp = missed[lang]
                    temp.append(video_id)
                    missed[lang] = temp

    miss = []
    for la in lang_list:
        miss.append(missed[la])
    res_u = set().union(*miss)
    vid_nlang = list(set(url)-set(res_u))

    miss = []
    for la in ['en','fr']:
        miss.append(missed[la])
    res_u = set().union(*miss)
    vid_enfr_lang = list(set(url)-set(res_u))

    miss = []
    for la in ['en','es']:
        miss.append(missed[la])
    res_u = set().union(*miss)
    vid_enes_lang = list(set(url)-set(res_u))

    miss = []
    for la in ['es','fr']:
        miss.append(missed[la])
    res_u = set().union(*miss)
    vid_esfr_lang = list(set(url)-set(res_u))

    with open( listofvid_path + "vid_en_es_fr_lang.txt", 'w') as the_file:
        for t in vid_nlang:
            the_file.write( str(t) + '\n' )

    with open( listofvid_path + "vid_en_fr_lang.txt", 'w') as the_file:
        for t in vid_enfr_lang:
            the_file.write( str(t) + '\n' )

    with open( listofvid_path + "vid_en_es_lang.txt", 'w') as the_file:
        for t in vid_enes_lang:
            the_file.write( str(t) + '\n' )

    with open( listofvid_path + "vid_es_fr_lang.txt", 'w') as the_file:
        for t in vid_esfr_lang:
            the_file.write( str(t) + '\n' )



if __name__ == "__main__":
    apikey = "" # enter google dev key


    oppath = '' # output path for srt files 
    listofvid_path = '' # output path for index file


    scrapyt(apikey,oppath,listofvid_path)
