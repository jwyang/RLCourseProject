local random = require 'common.random'
local custom_observations = require 'decorators.custom_observations'
local timeout = require 'decorators.timeout'
local make_map = require 'common.make_map'
local pickups = require 'common.pickups'

local BOT_NAMES_COLOR = {
    'CygniColor',
    'LeonisColor',
    'EpsilonColor',
    'CepheiColor',
    'CentauriColor',
    'DraconisColor'
}

local BOT_NAMES = {
    'Cygni',
    'Leonis',
    'Epsilon',
    'Cephei',
    'Centauri',
    'Draconis'
}

local factory = {}

--[[ Creates a Squirrel map API.
Keyword arguments:

*   `mapName` (string) - Name of map to load.
*   `botCount` (number, [-1, 6], default 4) - Number of bots. (-1 for all).
*   `skill` (number, [1.0, 5.0], default 4.0) - Skill level of bot.
*   `episodeLengthSeconds` (number, default 30) - Episode length in seconds.
*   `color` (boolean, default false) - Change color of bots each episode.
]]
function factory.createLevelApi(kwargs)
  kwargs.botCount = kwargs.botCount or 1
  kwargs.skill = kwargs.skill or 4.0
  kwargs.episodeLengthSeconds = kwargs.episodeLengthSeconds or 30
  kwargs.color = kwargs.color or false
  assert(kwargs.botCount <= (kwargs.color and #BOT_NAMES_COLOR or #BOT_NAMES))
  local api = {}
  api.finish_count = 2


  function api:start(episode, seed, params)
    random.seed(seed)
    if kwargs.color then
      -- Pick a random angle.
      api.bot_hue_degrees_ = random.uniformInt(0, 359)
    end
    api._count = 0
    --brought from our map
  end

  --brought from our map
  function api:nextMap()
    map = "A P I A P"
    for i = 0, api._count do
      map = map.." A"
    end
    api._finish_count = api._count + api._count
    api._count = 0
    return make_map.makeMap("demo_map_pred_" .. api._finish_count, map)
  end

  --brought from our map
  function api:commandLine(oldCommandLine)
    return make_map.commandLine(oldCommandLine)
  end
  --brought from our map
  function api:createPickup(className)
    return pickups.defaults[className]
  end

  function api:pickup(spawn_id)
    api._count = api._count + 1
    if api._count >= api._finish_count then
      game:finishMap()
      api._count = 0
    end
  end

  function api:addBots()
    local bots = {}
    for i, name in ipairs(kwargs.color and BOT_NAMES_COLOR or BOT_NAMES) do
      if i == kwargs.botCount + 1 then
        break
      end
      bots[#bots + 1] = {name = name, skill = kwargs.skill}
    end
    return bots
  end

  custom_observations.decorate(api)
  timeout.decorate(api, kwargs.episodeLengthSeconds)
  return api
end

return factory