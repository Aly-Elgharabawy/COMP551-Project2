overwatch_maps = ['2CP','Koth', 'Horizon Lunar Colony', 'Temple of Anubis','Volskaya','Volskaya Industries',
'Lijiang','Lijiang Tower','Ilios','Oasis','Watchpoint Gibraltar','Watchpoint: Gibraltar','Gibraltar','Dorado',
'Havana','Route 66','Junkertown','Rialto',"King's Row",'Kings Row','Numbani','Eichenwalde','Blizzard World','Overwatch']


overwatch_heroes = ['D.va','Dva','Orisa','Reinhardt','Rein','Roadhog','Sigma','Winston','Wrecking Ball','Hammond','Zarya',
'Ashe','Bastion','Doomfist','Genji','Hanzo','Junkrat','Mccree','Mei','Pharah','Reaper','Soldier 76','Soldier: 76',
'Sombra','Symmetra','Torbjorn','Torb','Torbjörn','Tracer','Widowmaker','Baptiste','Brigitte','Lucio','Moira','Zenyatta']

lol_champions = ['Aatrox','Ahri','Akali','Alistar','Amumu','Anivia','Ashe','Aurelion Sol','Asol','Azir','Bard','Blitzcrank','Blitz','Brand','Braum','Caitlyn','Cait','Camille',
'Cassiopeia',"Cho'Gath",'Cho','Corki','Darius','Diana','Dr. Mundo','Mundo','Draven','Ekko','Elise','Evelynn','Ezreal','Fiddlesticks','Fiora','Fizz','Galio','Gangplank','Garen',
'Gnar','Gragas','Graves','Hecarim','Heca','Heimer','Illaoi','Irelia','Ivern','Janna','Jarvan IV','J4','Jax','Jayce','Jhin','Jinx',"Kai'sa",'Kaisa','Kalista','Karthus','Kassadin',
'Kass','Katarina','Kata','Kayle','Kayn','Kennen','KhaZix','Kha','Kindred',"Kogmaw",'Kog',"Kog'Maw",'LeBlanc','Lee Sin','Leona','Lissandra','Lucian','Lulu','Lux','Malphite','Malph','Malz','Malzahar','Maokai','Master Yi',
'Miss Fortune','Mordekaiser','Mord','Morgana','Nami','Nasus','Nautilus','Neeko','Nidalee','Nocturne','Nunu', 'Willump','Olaf','Orianna','Ornn','Pantheon','Panth','Poppy','Pyke',
'Qiyana','Quinn','Rakan','Rammus',"Reksai",'Renekton','Rengar','Rango','Riven','Rumble','Ryze','Sejuani','Sej','Shaco','Shen','Shyvana','Singed','Sion','Sivir','Skarner','Sona','Soraka',
'Swain','Sylas','Syndra','Tahm Kench','Taliyah','Talon','Taric','Teemo','Thresh','Tristana','Trist','Trynd','Trundle','Tryndamere','Twisted Fate','Udyr','Urgot','Varus','Vayne',
'Veigar','VelKoz',"Vel'Koz",'Viktor','Volibear','Warwick','Wukong','Xayah','Xerath','Xin Zhao','Yasuo','Yorick','Yuumi','Ziggs','Zilean','Zyra']

lol_terms = ['midlan','toplan','mid lane','top lane','botlan','bot lane','Jungle','ADC','krugs','bush','lane','hook','lcs','guinsoo','zhonya','rageblade','ruined king','botrk',
'triforce','trinity force','runaan','riot','champion','raptors','baron','morello','rabadon','deathcap']

wow_terms = ['Burning Crusade','/9m','TBC','Lich King','Wotlk','Cataclysm','Cata','Pandaria','Draenor','Azeroth','Bfa','Illidan','Stormrage',
'Karazhan','Azshara','The Eternal Palace','Throne of Thunder','Siege of Orgrimmar','Garrosh','Blackrock','Icecrown','nighthold','firelands',
"Mogu'shan",'Emerald Nightmare','Ulduar','Uldaman','Blackwing','Deathwing','Sargeras',"Ahn'Qiraj",'AhnQiraj','Dragon Soul','Molten Core','Ragnaros'
,'Heart of Fear','Northrend','Mythic','M+','ilvl','Guild','lfr','Highmaul','Zandalar','Kul Tiras','Jaina','Sylvanas','Anduin',"Gul'dan",'Guldan','Antorus','Black Temple','Dazaralor',"Dazar'alor",'Uldir','Hyjal','Onyxia','Thrall','Arena','PvP','PvE','Warsong']

wow_classes = ['Druid','Hunter','Paladin','Pally','Priest','Rogue','Shaman','Warlock','Warrior','Death Knight','Monk','Demon Hunter']

wow_races = ['Draenei','Dwarf','Gnome','Night Elf','Pandaren','Worgen','Void elf','Lightforged','Dark Iron','Kul Tiran',
'Undead','Nightborne','Orc','Tauren','Troll','Highmountain',"Mag'har",'Maghar','Zandalari','Vulpera','Goblin','Blood Elf']

def create_features(comments):
    comments = list(comments)
    found = False
    ow_maps = []
    for comment in comments:
        for overwatch_map in overwatch_maps:
            if(overwatch_map.lower() in comment.lower()):
                ow_maps.append(1)
                found = True
                break
        if(found == False):
            ow_maps.append(0)
        found = False
    ow_heroes = []
    found = False
    for comment in comments:
        for overwatch_hero in overwatch_heroes:
            if(overwatch_hero.lower() in comment.lower()):
                ow_heroes.append(1)
                found = True
                break
        if(found == False):
            ow_heroes.append(0)
        found = False
    lol_champs = []
    found = False
    for comment in comments:
        for lol_champion in lol_champions:
            if(lol_champion.lower() in comment.lower()):
                lol_champs.append(1)
                found = True
                break
        if(found == False):
            lol_champs.append(0)
        found = False
    lol_t = []
    found = False
    for comment in comments:
        for lol_term in lol_terms:
            if(lol_term.lower() in comment.lower()):
                lol_t.append(1)
                found = True
                break
        if(found == False):
            lol_t.append(0)
        found = False
    wow_t = []
    found = False
    for comment in comments:
        for wow_term in wow_terms:
            if(wow_term.lower() in comment.lower()):
                wow_t.append(1)
                found = True
                break
        if(found == False):
            wow_t.append(0)
        found = False
    wow_c = []
    found = False
    for comment in comments:
        for wow_class in wow_classes:
            if(wow_class.lower() in comment.lower()):
                wow_c.append(1)
                found = True
                break
        if(found == False):
            wow_c.append(0)
        found = False
    wow_r = []
    found = False
    for comment in comments:
        for wow_race in wow_races:
            if(wow_race.lower() in comment.lower()):
                wow_r.append(1)
                found = True
                break
        if(found == False):
            wow_r.append(0)
        found = False

    #now features are made, make list of all
    new_features = [ow_maps,ow_heroes,lol_champs,lol_t,wow_t,wow_c,wow_r]
    return new_features