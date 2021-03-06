/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.commons.clilda;

import org.apache.commons.clilda.Option;
import org.apache.commons.clilda.OptionGroup;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Main entry-point into the library.
 * <p>
 * Options represents a collection of {@link org.apache.commons.clilda.Option} objects, which
 * describe the possible options for a command-line.
 * <p>
 * It may flexibly parse long and short options, with or without
 * values.  Additionally, it may parse only a portion of a commandline,
 * allowing for flexible multi-stage parsing.
 *
 * @see org.apache.commons.clilda.CommandLine
 *
 * @version $Id: Options.java 1685376 2015-06-14 09:51:59Z britter $
 */
public class Options implements Serializable
{
    /** The serial version UID. */
    private static final long serialVersionUID = 1L;

    /** a map of the options with the character key */
    private final Map<String, org.apache.commons.clilda.Option> shortOpts = new LinkedHashMap<String, org.apache.commons.clilda.Option>();

    /** a map of the options with the long key */
    private final Map<String, org.apache.commons.clilda.Option> longOpts = new LinkedHashMap<String, org.apache.commons.clilda.Option>();

    /** a map of the required options */
    // N.B. This can contain either a String (addOption) or an OptionGroup (addOptionGroup)
    // TODO this seems wrong
    private final List<Object> requiredOpts = new ArrayList<Object>();

    /** a map of the option groups */
    private final Map<String, org.apache.commons.clilda.OptionGroup> optionGroups = new HashMap<String, org.apache.commons.clilda.OptionGroup>();

    /**
     * Add the specified option group.
     *
     * @param group the OptionGroup that is to be added
     * @return the resulting Options instance
     */
    public Options addOptionGroup(org.apache.commons.clilda.OptionGroup group)
    {
        if (group.isRequired())
        {
            requiredOpts.add(group);
        }

        for (org.apache.commons.clilda.Option option : group.getOptions())
        {
            // an Option cannot be required if it is in an
            // OptionGroup, either the group is required or
            // nothing is required
            option.setRequired(false);
            addOption(option);

            optionGroups.put(option.getKey(), group);
        }

        return this;
    }

    /**
     * Lists the OptionGroups that are members of this Options instance.
     *
     * @return a Collection of OptionGroup instances.
     */
    Collection<org.apache.commons.clilda.OptionGroup> getOptionGroups()
    {
        return new HashSet<org.apache.commons.clilda.OptionGroup>(optionGroups.values());
    }

    /**
     * Add an option that only contains a short name.
     * The option does not take an argument.
     *
     * @param opt Short single-character name of the option.
     * @param description Self-documenting description
     * @return the resulting Options instance
     * @since 1.3
     */
    public Options addOption(String opt, String description)
    {
        addOption(opt, null, false, description);
        return this;
    }

    /**
     * Add an option that only contains a short-name.
     * It may be specified as requiring an argument.
     *
     * @param opt Short single-character name of the option.
     * @param hasArg flag signally if an argument is required after this option
     * @param description Self-documenting description
     * @return the resulting Options instance
     */
    public Options addOption(String opt, boolean hasArg, String description)
    {
        addOption(opt, null, hasArg, description);
        return this;
    }

    /**
     * Add an option that contains a short-name and a long-name.
     * It may be specified as requiring an argument.
     *
     * @param opt Short single-character name of the option.
     * @param longOpt Long multi-character name of the option.
     * @param hasArg flag signally if an argument is required after this option
     * @param description Self-documenting description
     * @return the resulting Options instance
     */
    public Options addOption(String opt, String longOpt, boolean hasArg, String description)
    {
        addOption(new org.apache.commons.clilda.Option(opt, longOpt, hasArg, description));
        return this;
    }

    /**
     * Adds an option instance
     *
     * @param opt the option that is to be added
     * @return the resulting Options instance
     */
    public Options addOption(org.apache.commons.clilda.Option opt)
    {
        String key = opt.getKey();

        // add it to the long option list
        if (opt.hasLongOpt())
        {
            longOpts.put(opt.getLongOpt(), opt);
        }

        // if the option is required add it to the required list
        if (opt.isRequired())
        {
            if (requiredOpts.contains(key))
            {
                requiredOpts.remove(requiredOpts.indexOf(key));
            }
            requiredOpts.add(key);
        }

        shortOpts.put(key, opt);

        return this;
    }

    /**
     * Retrieve a read-only list of options in this set
     *
     * @return read-only Collection of {@link org.apache.commons.clilda.Option} objects in this descriptor
     */
    public Collection<org.apache.commons.clilda.Option> getOptions()
    {
        return Collections.unmodifiableCollection(helpOptions());
    }

    /**
     * Returns the Options for use by the HelpFormatter.
     *
     * @return the List of Options
     */
    List<org.apache.commons.clilda.Option> helpOptions()
    {
        return new ArrayList<org.apache.commons.clilda.Option>(shortOpts.values());
    }

    /**
     * Returns the required options.
     *
     * @return read-only List of required options
     */
    public List getRequiredOptions()
    {
        return Collections.unmodifiableList(requiredOpts);
    }

    /**
     * Retrieve the {@link org.apache.commons.clilda.Option} matching the long or short name specified.
     * The leading hyphens in the name are ignored (up to 2).
     *
     * @param opt short or long name of the {@link org.apache.commons.clilda.Option}
     * @return the option represented by opt
     */
    public org.apache.commons.clilda.Option getOption(String opt)
    {
        opt = org.apache.commons.clilda.Util.stripLeadingHyphens(opt);

        if (shortOpts.containsKey(opt))
        {
            return shortOpts.get(opt);
        }

        return longOpts.get(opt);
    }

    /**
     * Returns the options with a long name starting with the name specified.
     * 
     * @param opt the partial name of the option
     * @return the options matching the partial name specified, or an empty list if none matches
     * @since 1.3
     */
    public List<String> getMatchingOptions(String opt)
    {
        opt = org.apache.commons.clilda.Util.stripLeadingHyphens(opt);
        
        List<String> matchingOpts = new ArrayList<String>();

        // for a perfect match return the single option only
        if (longOpts.keySet().contains(opt))
        {
            return Collections.singletonList(opt);
        }

        for (String longOpt : longOpts.keySet())
        {
            if (longOpt.startsWith(opt))
            {
                matchingOpts.add(longOpt);
            }
        }
        
        return matchingOpts;
    }

    /**
     * Returns whether the named {@link org.apache.commons.clilda.Option} is a member of this {@link Options}.
     *
     * @param opt short or long name of the {@link org.apache.commons.clilda.Option}
     * @return true if the named {@link org.apache.commons.clilda.Option} is a member of this {@link Options}
     */
    public boolean hasOption(String opt)
    {
        opt = org.apache.commons.clilda.Util.stripLeadingHyphens(opt);

        return shortOpts.containsKey(opt) || longOpts.containsKey(opt);
    }

    /**
     * Returns whether the named {@link org.apache.commons.clilda.Option} is a member of this {@link Options}.
     *
     * @param opt long name of the {@link org.apache.commons.clilda.Option}
     * @return true if the named {@link org.apache.commons.clilda.Option} is a member of this {@link Options}
     * @since 1.3
     */
    public boolean hasLongOption(String opt)
    {
        opt = org.apache.commons.clilda.Util.stripLeadingHyphens(opt);

        return longOpts.containsKey(opt);
    }

    /**
     * Returns whether the named {@link org.apache.commons.clilda.Option} is a member of this {@link Options}.
     *
     * @param opt short name of the {@link org.apache.commons.clilda.Option}
     * @return true if the named {@link org.apache.commons.clilda.Option} is a member of this {@link Options}
     * @since 1.3
     */
    public boolean hasShortOption(String opt)
    {
        opt = org.apache.commons.clilda.Util.stripLeadingHyphens(opt);

        return shortOpts.containsKey(opt);
    }

    /**
     * Returns the OptionGroup the <code>opt</code> belongs to.
     * @param opt the option whose OptionGroup is being queried.
     *
     * @return the OptionGroup if <code>opt</code> is part
     * of an OptionGroup, otherwise return null
     */
    public OptionGroup getOptionGroup(Option opt)
    {
        return optionGroups.get(opt.getKey());
    }

    /**
     * Dump state, suitable for debugging.
     *
     * @return Stringified form of this object
     */
    @Override
    public String toString()
    {
        StringBuilder buf = new StringBuilder();

        buf.append("[ Options: [ short ");
        buf.append(shortOpts.toString());
        buf.append(" ] [ long ");
        buf.append(longOpts);
        buf.append(" ]");

        return buf.toString();
    }
}
