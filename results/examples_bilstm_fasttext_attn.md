# Examples for bilstm_fasttext_attn

## Correct, highest confidence

| tweet | gold | pred | conf |
|---|---|---|---|
| @user ISABUKURU NZIZA NYAKUBAHWA. IMANA YAREMYE U #Rwanda N'ISI YOSE IKOMEZE IKUTURINDIRE, IKONGERE IMBARAGA, UBUHANGA N'UBUTWARI BIGUSHOBOZA KURUZAMURA IZINDI NTERA NZIZA. #RwOT | positive | positive | 1.00 |
| "Ubutwari mu banyarwanda, agaciro kacu."""" @user tubifurije umunsi mwiza w intwari ku banyarwanda bose n inshuti zacu #Ubutwari2022 #SMVDrive https://t.co/oPEylfNQr6" | positive | positive | 1.00 |
| @user @user Turagushimira umuhate n'umurava wakoranye akazi wari ushinzwe, Nahandi hose uzajya uzakomeze muriyo ntero. Imana iduhe umugisha | positive | positive | 0.99 |
| Inkotanyi ni Ubuzima. Umunsi mwiza wo #Kwibohora27 kuri twese. Umunsi nk’uyu utwibutsa Agaciro Twasubijwe n’Inkotanyi: Tugasigasire; Imihigo irakomeje... https://t.co/ItyYDLvhf2 | positive | positive | 0.99 |
| Chorale Illuminatio yifurije aba Kristu bose muri Rusange umunsi mwiza w'icyumweru cya Pentecôte. Dukomeze kuzirikana ingabire za Roho Mutagatifu 🙏 https://t.co/9BgMfbciMn | positive | positive | 0.99 |
| @user ndabashimiye umwanya mudahwema impano z'Abanyarwanda nange @user ndimo. Mboneyeho n'Umwanya wo gushimira @user wantumiye mukiganiro DUNDA SHOW. ndashimira kandi @user @user ndetse na @user https://t.co/96UII9TlDb | positive | positive | 0.99 |
| "Umunsi mwiza w'Intwari z'Igihugu """"Ubutwari mu banyarwanda. Agaciro Kacu. https://t.co/ukuDlqZzSJ" | positive | positive | 0.99 |
| @user: Turifuza ko muba abayobozi bafite urukundo; urukundo rwubaka icyizere, nacyo kikubaka ubufatanye, ubufatanye bugafasha kugera ku ntego. #UmuturageKuIsonga https://t.co/ERG6slhuxa | positive | positive | 0.99 |
| Banyarwanda, namwe mukunda u Rwanda, ni muze duhuze imbaraga dutsinde umwanzi #COVID19 uruhare rwanjye nawe ningenzi #Ndindankurinde tuzatsinda. https://t.co/LXqtCVEEyr | positive | positive | 0.99 |
| Aba hano mwese Mugire umunsi mwiza, Ntihagire ugusaba ubufasha ubishoboye ugire icyo ukora.Amahoro | positive | positive | 0.99 |
| Umutima wanjye wuzuye ibyishimo n’umunezero utagira uko ungana kubera Ubu butumwa bwawe muvandimwe @user ♥️ 💜 ♥️ Nzakomeza guharanira ibyiza byose kuri bose 🙏🏿 https://t.co/TO0zdSeJr9 | positive | positive | 0.99 |
| Umunsi mwiza @user iyo wakoze neza,ukaruhuka neza,undi munsi urushaho kunoza umurimo https://t.co/Vx1fpWvv2g | positive | positive | 0.99 |

## Wrong, highest confidence (most 'confident mistakes')

| tweet | gold | pred | conf | agreement |
|---|---|---|---|---|
| @user Mwiriwe neza ndimuhanga nibyiza kuba Aho irubavu baresheje imihigo bagafatanya bakizigamira Kandi dukomeze twubahirize Amabwiriza yo kwirinda covide 19 | neutral | positive | 0.98 | 2.0 |
| Hanyuma muri ya gahunda yo gushyigikira #Abacu nutekereza Imyambaro ya siporo igezweho ntuzarenze ingohe #IwacuSportsWear 💙💛💚🇷🇼🇷🇼🇷🇼 Cc. @user @user https://t.co/08YLueCSCi | neutral | positive | 0.95 | 2.0 |
| Breaking News: Louis Rurangirwa avuze ko hari itegeko ritubahirijwe, ahagaritse kwiyamamaza. #FERWAFAElections2021 @user @user 95.3 FM https://t.co/MVTy2fsCBS | negative | neutral | 0.94 | 1.0 |
| Dushima umuhate mushyira mukurwanya icyi cyorezo dufatanyije byose birashoboka, dore uko Rwerere sector, Ruconsho cell byaribimeze. #Ntakudohoka https://t.co/Rq9pPZbe4x https://t.co/UBEM1i3OWx | neutral | positive | 0.94 | 2.0 |
| @user @user @user @user @user @user @user @user Nta kibazo na kimwe. Ikaze kucyumweru gitaha kuva 6-7h30am | positive | neutral | 0.93 | 2.0 |
| Menya ubutwari bw'abatangije urugamba rwo #kubohora u Rwan da... https://t.co/AfMkwylgei | neutral | positive | 0.91 | 2.0 |
| Ntimucikwe umukino wa 1/2 w'Igikombe cy'Amahoro uhuza Rayon Sports na APR FC. Ni kuri @user mu kanya 15h00. https://t.co/3I3bYpifpg | positive | neutral | 0.91 | nan |
| Imig 3:5 [5]Wiringire Uwiteka n'umutima wawe wose,We kwishingikiriza ku buhanga bwawe. https://t.co/y9OUBX84Sb | neutral | positive | 0.90 | 2.0 |
| Uyu munsi Urubyiruko rwabakorerabushake ba @user mungamba zo gufasha abaremye isoko mugukomeza kwirinda icyorezo cya covid19 @user @user @user @user https://t.co/WxEZFPfJKP | neutral | positive | 0.88 | 2.0 |
| kuki twishimira umunsi w'ubwigenge Ariko ntimutubwire uwaharaniye ubwo bwigenge?? 👁👁👁👁🙌🙌🙌 | negative | positive | 0.88 | 3.0 |
| @user Nakwirukana umugore we uryamana nutaramukoye, kuko murumuna wanjye nabamubaza iki bamuhaye ibiryo akirira🤗🤗 | neutral | negative | 0.87 | nan |
| Imyitozo ya mbere y’umukino yagenze neza #AsKigaliVsDCMP #CAFConfederationCup https://t.co/gJMq6IETCp | positive | neutral | 0.87 | 2.0 |
| @user Niyo muhadjiri agitsinda yari kuvuga ko 10-4=6 this is shit mn | negative | neutral | 0.87 | 2.0 |
| 🚨AMAKURU MASHYA🚨 Inama y'Abaminisitiri yatangaje ko bitakiri itegeko kwambara agapfukamunwa. #RBAAmakuru https://t.co/dP9HCHHTpf | positive | neutral | 0.86 | nan |
| @user @user @user @user @user Utyoooo. Umugabo w'ukuri yubaha umugore. Umugore w'ukuri yubaha umugabo. Ibyo birahagije ngo batengamare. | positive | negative | 0.86 | 2.0 |
| @user @user Gutera inda, byumvikana nko gutanga umwana kuko iyo nda ni yo abyara. Ntabwo ayimutera, bayikora bombi. | neutral | negative | 0.86 | 2.0 |
| Niba wumva ushishikajwe niterambere ry'umuhinzi wibirayi dushyigikire ukushoboye mubuhinzi bwibirayi byera vuba Kandi bidatewe umuti https://t.co/jnLPBAuPsJ | neutral | positive | 0.86 | 2.0 |
| Inyubako y'umurenge wa @user yatwaye amafaranga y'u Rwanda miliyoni 900 gusa, nyamara ni imwe mu nyubako zigezweho dufite. @user 👏🏾👏🏾👏🏾 @user @user | positive | neutral | 0.85 | 3.0 |
| Ese koko birakwiye ko umubyeyi abura ubuzima mugiye atanga ubundi (abyara)??? Sobanukirwa impanvu zatuma umubyeyi abura ubuzima mugiye abyara👇 https://t.co/UG55ES2l5x https://t.co/oPwAf3d9Ya | neutral | negative | 0.85 | 2.0 |
| @user @user Uyu musore yaranyandikiye Ambwira ngo imbwa yimana Irigende ngirango arantutse Aho kumbe niko yitwa😂😂😂😂😂 Sinamenye ko ariyo mazinaye😂😂 | neutral | negative | 0.85 | 2.0 |
| @user @user @user @user Hagize uyigobotora asigarana ubwoba bwo kugira umuyaga wo mu nda, uruvugiro akarubura da, agasa n’udafite aho aca ingando. | neutral | negative | 0.85 | 2.0 |
| @user Gus buriya ninkikibazo jyanibaza burigihe bavuga ngo bafashe inzoga zinkorano kuki batarafata fanta zinkorano 😂🤔🤔 | neutral | negative | 0.84 | nan |
| @user Hhhhhhhhhh wamuntuwe wasanga aba amurinda imibu akana yicara ngira kugirango mwokumva urusaku no kuduhira kwayo | positive | negative | 0.84 | 2.0 |
| Kuvugirinka Mubukwe uzi neza ko wakoye amafaranga ninko kubwira umugore wabuze urubyaro ngo azane umwanana winsina muwite izina. | neutral | negative | 0.84 | 2.0 |

## Wrong, random sample

| tweet | gold | pred | conf | agreement |
|---|---|---|---|---|
| @user @user Hahaha video za gospel kuzifata bigombera kuba ufite umugore se? | neutral | negative | 0.52 | 2.0 |
| Ariko iyi kipe niba ari uko ifite #igikundiro, Fédération namwe murayikunda mugakanya https://t.co/vc4ZkcFoFo | positive | neutral | 0.53 | 2.0 |
| @user Nakwirukana umugore we uryamana nutaramukoye, kuko murumuna wanjye nabamubaza iki bamuhaye ibiryo akirira🤗🤗 | neutral | negative | 0.87 | nan |
| kuki twishimira umunsi w'ubwigenge Ariko ntimutubwire uwaharaniye ubwo bwigenge?? 👁👁👁👁🙌🙌🙌 | negative | positive | 0.88 | 3.0 |
| @user @user @user Bagirishya nambwira ko umurozi yashyinguwe ndatangira kwitegura! | negative | neutral | 0.48 | 2.0 |
| @user @user Gutera inda, byumvikana nko gutanga umwana kuko iyo nda ni yo abyara. Ntabwo ayimutera, bayikora bombi. | neutral | negative | 0.86 | 2.0 |
| @user na @user muze mumpe Congs ndyame! 😂 | positive | neutral | 0.66 | 3.0 |
| @user @user @user @user Mva ku izima gute Kandi ndi mu kuri?? | positive | neutral | 0.54 | 2.0 |
| @user aratabaza kuri uyu munsi w'intwari, kuko ari gukorerwa akarengane (ihohoterwa) n'umuyobozi wa @user, arasaba ababifitemo ububasha kumufasha akarenganurwa. https://t.co/sUL6MtAhnF | negative | neutral | 0.39 | 0.0 |
| @user @user Kuki batinya kuvuga ‘oya ‘ cg ‘yego ‘ ako kanya ko bifasha umusore gushakisha ikindi ikiganza mu ituze? | neutral | negative | 0.73 | 2.0 |
| @user Kuki wumva ko ibicuruzwa byo Mugihugu cyawe byahenduka ariko ibyahandi ukumva ko ntakibazo byose ni STANDARD my guy | neutral | negative | 0.67 | 2.0 |
| @user @user Yarahari x ngo aduhe amafoto cg video ? | neutral | negative | 0.48 | 3.0 |
| #runda muri gare abakorerabushake bari gufasha mukubahiriza amabwiriza yatanzwe yo kwirinda covid19 @user https://t.co/GOSVP2rBFF | positive | neutral | 0.77 | 2.0 |
| @user @user @user @user @user @user @user @user Ubundi bafite inyogo yabo | neutral | negative | 0.49 | 2.0 |
| Amafaranga n'ububasha birikongera individualism mu Rwanda @user .Harigukorwa iki mu Rwanda ubusumbane bwere kwiyongera? | positive | neutral | 0.50 | 2.0 |
| @user Ni igihe Inka ziba zimaze kurisha kumanywa hanyuma zikaryama zirimo ziruhuka. | neutral | negative | 0.61 | 2.0 |
| @user @user @user @user @user @user @user @user None se ko iy'uku kwezi nayibuze nta kuntu mwabaza @user ko akunze kubimenya cyane cyane mu mpera zukwezii🤦 | neutral | negative | 0.59 | 3.0 |
| @user Rimwe na rimwe mba mbona ufite ubwenge... nawe watorokanye cash za RNP ubutabera buragutegereje. | positive | negative | 0.49 | 2.0 |
| Hanyuma muri ya gahunda yo gushyigikira #Abacu nutekereza Imyambaro ya siporo igezweho ntuzarenze ingohe #IwacuSportsWear 💙💛💚🇷🇼🇷🇼🇷🇼 Cc. @user @user https://t.co/08YLueCSCi | neutral | positive | 0.95 | 2.0 |
| @user @user Biterwa n'ureba, ariko ni mwiza! Anyway, Merci! | positive | neutral | 0.42 | 2.0 |
| Ntimucikwe umukino wa 1/2 w'Igikombe cy'Amahoro uhuza Rayon Sports na APR FC. Ni kuri @user mu kanya 15h00. https://t.co/3I3bYpifpg | positive | neutral | 0.91 | nan |
| @user @user Ese ko wameze amababa raa ? 😃😃 | negative | neutral | 0.55 | 2.0 |
| @user Dufite ubwoko bwawe nawe. Utuvugishe kuri 0784573221 tuguhe ibisobanuro birambuye | positive | neutral | 0.46 | nan |
| umukobwa twararanye arimuri douches arikuririmba ngo ngwino urebe ibyiza by'urwanda ubuse njyeyo koko??? | neutral | negative | 0.65 | nan |