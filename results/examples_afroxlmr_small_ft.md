# Examples for afroxlmr_small_ft

## Correct, highest confidence

| tweet | gold | pred | conf |
|---|---|---|---|
| Dushimye Imana yo ishobora byose,kuba yemeye ko turamuka...Iragahora ku ngoma ubu n'iteka.Amahoro mwese bavandimwe!! | positive | positive | 0.99 |
| @user @user Mwishyuke Benimana kandi mugumemo. Muhorane ibyiza mu rwanyu. | positive | positive | 0.99 |
| @user Gusa amahoro allah asakaza kube akugereho wese nawe munywanyi mwiza🤲🤲🤗🤗🤗 | positive | positive | 0.99 |
| Amahoro mwese bantu ba hano,Uyu munsi Imana ibiteho murusheho kugirirwa neza na Nyagasani.Yellow Paper ,amasaha yongerewe,duhahe turonke. | positive | positive | 0.99 |
| @user Mbega weee! Ubu nari hehe! Nguhaye umugisha w’ababyeyi. Muzagire urugo ruhire wowe n’uwawe. Urubyaro, ubutunzi no gutunganirwa mbibaturiyeho. | positive | positive | 0.99 |
| Ikaze mu bawe bagukunda🙏🏾 Komeza urushakire imbuto n'amaboko. Blessings https://t.co/JH39i3ObdT | positive | positive | 0.99 |
| Umutima wanjye wuzuye ibyishimo n’umunezero utagira uko ungana kubera Ubu butumwa bwawe muvandimwe @user ♥️ 💜 ♥️ Nzakomeza guharanira ibyiza byose kuri bose 🙏🏿 https://t.co/TO0zdSeJr9 | positive | positive | 0.99 |
| @user Uracyeye muhungu wanjye! Uyu mwaka Uwiteka akomeze agutotobeke wakire amahanga yose! | positive | positive | 0.99 |
| "Ubutwari mu banyarwanda, agaciro kacu."""" @user tubifurije umunsi mwiza w intwari ku banyarwanda bose n inshuti zacu #Ubutwari2022 #SMVDrive https://t.co/oPEylfNQr6" | positive | positive | 0.99 |
| @user @user @user @user @user @user @user Masafi, Mashaba, Munguiko rwose Imana ikugwirize ibyiza. Isabukuru nziza. Iyi COVID-19 izashyira itubise | positive | positive | 0.99 |
| @user @user @user Amahoro nawe wa mfura we. Urukumbuzi rumarwa no kurumara uzaze turikibitire ahirengeye! | positive | positive | 0.99 |
| Igitondo kiza ndakwifuriza kugira Isabato nziza kubayemera uyu munsi namwe mugiye gushabika muhahe muronke abatameze neza Mwihangane Imana irabazi | positive | positive | 0.99 |

## Wrong, highest confidence (most 'confident mistakes')

| tweet | gold | pred | conf | agreement |
|---|---|---|---|---|
| Imig 3:5 [5]Wiringire Uwiteka n'umutima wawe wose,We kwishingikiriza ku buhanga bwawe. https://t.co/y9OUBX84Sb | neutral | positive | 0.99 | 2.0 |
| Niba wumva ushishikajwe niterambere ry'umuhinzi wibirayi dushyigikire ukushoboye mubuhinzi bwibirayi byera vuba Kandi bidatewe umuti https://t.co/jnLPBAuPsJ | neutral | positive | 0.99 | 2.0 |
| @user Mwiriwe neza ndimuhanga nibyiza kuba Aho irubavu baresheje imihigo bagafatanya bakizigamira Kandi dukomeze twubahirize Amabwiriza yo kwirinda covide 19 | neutral | positive | 0.99 | 2.0 |
| #Babyeyi, Konsa umwana wawe neza n'ingenzi, kuko bituma akurana ubudahangarwa mu mubiri kandi bikamurinda kurwaragurika bya hato nahato. #breastfeedingweek2021 https://t.co/GXTDooLqZi | neutral | positive | 0.99 | 2.0 |
| Muri iki gitondo, Nsuhuje Abantu mwese muri muri Repubulika Yigenga Ya Kivu. Abaturanyi banyu #RDCongo, #Tanzania #Burundi #Rwanda #Ouganda turabatashya cyane. https://t.co/VdXBnZoHnS | neutral | positive | 0.99 | 2.0 |
| Dushima umuhate mushyira mukurwanya icyi cyorezo dufatanyije byose birashoboka, dore uko Rwerere sector, Ruconsho cell byaribimeze. #Ntakudohoka https://t.co/Rq9pPZbe4x https://t.co/UBEM1i3OWx | neutral | positive | 0.98 | 2.0 |
| Arata umwanya, kuko ibye birazwi ku buryo buhagije. Amateka ye mabi n'uyo yabloka abantu bose ntiyasibangana. https://t.co/THzOZgvxd2 | negative | positive | 0.98 | 2.0 |
| @user Imyitozo irakomeza iki cyumweru cyose,Gahunda ni yayindi rwose @user #IbyishimoNiYoNtego | neutral | positive | 0.98 | 3.0 |
| @user @user Ni nk'aho abagabo bagomba gushimisha abagore ariko abagore nta nshingano yo kwitura ineza | neutral | positive | 0.98 | nan |
| Hanyuma muri ya gahunda yo gushyigikira #Abacu nutekereza Imyambaro ya siporo igezweho ntuzarenze ingohe #IwacuSportsWear 💙💛💚🇷🇼🇷🇼🇷🇼 Cc. @user @user https://t.co/08YLueCSCi | neutral | positive | 0.98 | 2.0 |
| @user @user Nawe singushira amakenga uroye! | negative | positive | 0.98 | 2.0 |
| Murararitswe kuri uyu munsi hamwe na @user dufatanyirije hamwe twatsinda #ihohoterwa Ni #indaziterwaabangavu. @user @user @user @user @user @user @user @user https://t.co/yNXjIEVMQF | neutral | positive | 0.97 | 2.0 |
| @user @user @user Igihe cyo kuganira ku masezerano ndifuza kuhaba nka manager w’umukinnyi kuko hari amategeko menshi atazi! | positive | neutral | 0.97 | 2.0 |
| Ase Yago na Phil Peter bigeze he? Ka kana karacyiriza nka mbere? | negative | neutral | 0.97 | 2.0 |
| @user @user @user Bazabikorere muri gereza, uzagaragarwaho ingengabitekerezo yaba yarasohotse cg akirimo itegeko rizakurikizwa. | negative | neutral | 0.97 | 2.0 |
| @user Imibare nubundi yatuzonze kuva kera 😂😂😂😂😂 | negative | neutral | 0.97 | 2.0 |
| @user Hanyuma se izo ni inka muzakama!?😏😏 ariko rata nimwijyanire wenda beer muza mukanywa zagabanuka ku isoko ntizihende😂 | neutral | negative | 0.97 | nan |
| @user Sha ndi mu cyumba guhera nijoro, gusa umutwe uracyandya! 😭 Ngiye gufata pause nubura ingana n'umwaka. 😭 | negative | neutral | 0.97 | 2.0 |
| umukobwa twararanye arimuri douches arikuririmba ngo ngwino urebe ibyiza by'urwanda ubuse njyeyo koko??? | neutral | negative | 0.97 | nan |
| @user Hi Leah amakur ya kabanyana | positive | neutral | 0.96 | 2.0 |
| Mubusanzwe kugirango wamuryango mukuru wanjye @user ukugereho neza nuko nawe ugambana #ubugambanyi @user @user Nyarugenge, Kigali, Rwanda https://t.co/VDwQcDQ5Q5 | negative | positive | 0.96 | nan |
| @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user Ozeya umunsi twongeye gupanga guhura uzitwaze indangamuntu. Ubanza nkwitiranya nabona ibiro byawe nkagira ngo n’imyaka ni uko! | neutral | negative | 0.96 | 2.0 |
| @user Arasoza agira ati: fino mavubi alla fine😂😂 | neutral | positive | 0.96 | 3.0 |
| AMAKURU MASHYA: Uwari umuyobozi mukuru w’ikigo @user Bwana @user yahagaritswe ku mirimo ye kugira ngo akuriranwe kubyo ashinjwa. https://t.co/d7Yp0K9nCk | negative | neutral | 0.96 | 2.0 |

## Wrong, random sample

| tweet | gold | pred | conf | agreement |
|---|---|---|---|---|
| Umva uko ikibazo giteye mu buryo burambuye https://t.co/MX9YyutMTx #Rwanda @user @user @user @user @user @user @user @user @user @user @user @user @user @user | neutral | positive | 0.73 | 3.0 |
| Hanyuma muri ya gahunda yo gushyigikira #Abacu nutekereza Imyambaro ya siporo igezweho ntuzarenze ingohe #IwacuSportsWear 💙💛💚🇷🇼🇷🇼🇷🇼 Cc. @user @user https://t.co/08YLueCSCi | neutral | positive | 0.98 | 2.0 |
| Nukuri ibyo yakoze yaratumwe,none iyo ataza kujyayo byari bumugendekere gute ? Nyamara intumwa ntiyicwa! https://t.co/Mt7nljYiRq | positive | negative | 0.67 | 2.0 |
| @user Nibakomereze aho, ntawiyumvagamo uburinzi n'ubudahangarwa muri bo nk'umuyobozi wabo muri MRCD, cg umuyobozi wa FLN Irategeka.🤗 | positive | neutral | 0.62 | 2.0 |
| @user Ko U20 women volleyball twakubiswe nkabadahari, twaziziki bwana Guss? | negative | neutral | 0.77 | 3.0 |
| Reka twizere ko ba mubyeyi bakumva bakaguhoza amarira. Pole. @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user @user #Rwanda https://t.co/DYZIXOxfZB | neutral | positive | 0.93 | 2.0 |
| @user Aho nta foto idasanzwe washyize kuri Twitter? Hari igihe baguhana for some times kubera kurenga ku mabwiriza runaka. | neutral | negative | 0.62 | 2.0 |
| @user Hoya bihinduye isura babajyane kuri stade amahoro tubatore tunabareba dii bari kutuyobya wna @user @user @user | negative | positive | 0.45 | 2.0 |
| Niba wumva ushishikajwe niterambere ry'umuhinzi wibirayi dushyigikire ukushoboye mubuhinzi bwibirayi byera vuba Kandi bidatewe umuti https://t.co/jnLPBAuPsJ | neutral | positive | 0.99 | 2.0 |
| @user @user Zizane twisekere ni hahandi h'abazikora nawe byakubaho 🤪🤪🤪 | positive | neutral | 0.92 | 2.0 |
| @user Uwumva atabishaka ya retirinze se🤷 | negative | neutral | 0.52 | 3.0 |
| Uyu munsi umukobwa uterekanye inzara ze kuri IG ni Boniface 😂😎 | negative | neutral | 0.94 | 2.0 |
| @user @user Ngabo yanyoheje ngo ntitubivuge! 😂 | neutral | negative | 0.61 | 2.0 |
| Kurura hasi tugende umutima nkute mu Mari Tugende nkujyane tugende nkugeze iyo mu badage | negative | positive | 0.57 | 2.0 |
| @user @user @user 😅😅 uyu munsi ntarindi shaza nareba nge | positive | neutral | 0.92 | 2.0 |
| Umuhanzi mwiza cyane Niyo Bosco agiye gushyira hanze indirimbo yitwa Ishyano abakunzi be kumbugankoranyambaga bakomeje kugaragaza ko bayinyotewe. https://t.co/GrwJzqghMb | positive | neutral | 0.88 | 3.0 |
| @user @user @user Arifata agashyiraho umuntu wananiwe gusimbuka urukuramende rwa santimetero 50 Koko?😏😏🤣😂 | negative | neutral | 0.62 | 2.0 |
| Amahirwe yo gutsinda aracyahari🙌🙌 #ITA Goal⚽⚽⚽⚽ https://t.co/OifMh5YhAh | positive | neutral | 0.89 | 3.0 |
| #runda muri gare abakorerabushake bari gufasha mukubahiriza amabwiriza yatanzwe yo kwirinda covid19 @user https://t.co/GOSVP2rBFF | positive | neutral | 0.88 | 2.0 |
| @user Arasoza agira ati: fino mavubi alla fine😂😂 | neutral | positive | 0.96 | 3.0 |
| @user muri Ndera Hari ikibazo cya Network nubwo tuvuga mukatwihorera guhamagarana n'ikibazo! | neutral | negative | 0.84 | 2.0 |
| Ase Yago na Phil Peter bigeze he? Ka kana karacyiriza nka mbere? | negative | neutral | 0.97 | 2.0 |
| Niyegure nubundi asa nkuwasinyaga ibyo atabanje gusoma ubwose aho hasi nabaki yavuze amenyesha. https://t.co/WSQQwxMojo https://t.co/dKNhC3IaGE | negative | neutral | 0.62 | 2.0 |
| @user @user @user “...amucinya urushyi “..Byaje kugenda gute se nyuma y’aho? Barakomezanyije se? | negative | neutral | 0.67 | 2.0 |